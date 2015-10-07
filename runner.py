import concurrent.futures
import subprocess

V = list(range(0, 500, 25))

def run(voltage):
    subprocess.call(["python", "esq.py", "--Vesq", str(voltage)])
    return voltage

if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for v in executor.map(run, V):
            print("finished Vesq =", v)
