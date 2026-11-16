"""Record probe tests and the guarded main CPU suite as ASCII."""
from pathlib import Path
import subprocess


def main():
    commands = [
        ['.venv-probe/Scripts/python.exe','-m','unittest','discover','-s','v2/tests','-t','.','-p','test_probe_*.py'],
        # The literal required command was run and is preserved in WS2-tests.txt.
        # WS1's parser test reads the forbidden official file; future full-suite
        # validation must use the scoped read guard instead of repeating that.
        ['.venv/Scripts/python.exe','-m','v2.probe.safe_discovery'],
    ]
    codes=[]
    with Path('workspace_logs/WS2-tests-guarded.txt').open('w',encoding='ascii') as log:
        for command in commands:
            log.write('COMMAND: '+' '.join(command)+'\n')
            executable=str(Path(command[0]).resolve())
            result=subprocess.run([executable,*command[1:]],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            output=result.stdout.decode('utf-8',errors='replace').encode('ascii',errors='backslashreplace').decode('ascii')
            log.write(output)
            log.write('EXIT: '+str(result.returncode)+'\n')
            log.flush()
            print(output[-250:],flush=True)
            codes.append(result.returncode)
    if any(codes):
        raise SystemExit(1)


if __name__=='__main__':
    main()
