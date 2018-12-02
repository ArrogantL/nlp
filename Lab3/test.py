import time
import progressbar


if __name__ == '__main__':
    p = progressbar.ProgressBar()
    N = 1000
    p.start()
    for i in range(N):
        time.sleep(0.01)
        p.update(i + 1)
    p.finish()
