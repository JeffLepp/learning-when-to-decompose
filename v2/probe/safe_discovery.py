"""Run the full CPU suite while skipping any forbidden official-file read."""
from pathlib import Path
import unittest
from unittest.mock import patch

from v2.probe.extract import ROOT


def main():
    forbidden=(ROOT/'data/SCAN/length_split/tasks_test_length.txt').resolve()
    original=Path.open
    blocked=[]

    def guarded_open(path,*args,**kwargs):
        if path.resolve()==forbidden:
            blocked.append(str(path))
            raise unittest.SkipTest('WS2 hard rule: official length-test file must not be read')
        return original(path,*args,**kwargs)

    with patch.object(Path,'open',guarded_open):
        suite=unittest.defaultTestLoader.discover('v2/tests',top_level_dir='.')
        result=unittest.TextTestRunner(verbosity=1).run(suite)
    print('Forbidden file reads prevented:',len(blocked),flush=True)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__=='__main__':
    main()
