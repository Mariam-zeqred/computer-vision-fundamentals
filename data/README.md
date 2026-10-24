# Data

`coffee.png` (600 × 400, RGB) is the "coffee" sample image distributed with
[scikit-image](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.coffee).
Photograph by Rachel Michetti, courtesy of Pikolo Espresso Bar, released under
CC0 (no copyright restrictions). It was copied here unchanged so the lessons run
without downloading anything.

Later lessons that need more images say where they come from in this file.

`belt_synthetic.png` (640 × 480, RGB) is **synthetic**: `lesson2.py`'s
`make_belt()` draws twelve ellipses on a textured belt under a left-to-right
lighting gradient, adds dirt specks, Gaussian noise and 1 % dead or saturated
pixels, from a fixed seed, and rewrites this file on every run. It exists so
lesson 2's egg counter can be scored against a known answer. Nothing in it is a
photograph.

`belt_oblique_t0.png` and `belt_oblique_t1.png` (640 × 480, RGB) are **synthetic**
too: `lesson3.py` takes the belt above in two frames 37 px of belt travel apart
(the second with fresh sensor noise) and warps each with a chosen camera
homography so the belt appears as a trapezoid seen from an angle. They exist so
lesson 3's belt-travel measurement can be scored against a known answer. The
camera, its markers and the travel are all invented; the files are rewritten on
every run.

Lesson 4 writes no files. Its 5 000 training and validation patches and the
frame with three labels and two blobs are rendered in memory by `lesson4.py`
(`make_patch_set()` and `lesson2.make_belt(decorate=lesson4.distractors)`) from
fixed seeds, with the same rules as the belt above. They are synthetic too.

Lesson 5 uses **Fashion-MNIST** (Zalando Research, 2017; Han Xiao, Kashif Rasul
and Roland Vollgraf, "Fashion-MNIST: a Novel Image Dataset for Benchmarking
Machine Learning Algorithms", arXiv:1708.07747), released under the MIT licence:
70 000 real 28 × 28 grayscale photographs of clothing in ten classes, 60 000 for
training and 10 000 that the authors hold out as the test set. The four gzipped
IDX files (30 MB) are **not committed**; `lesson5.py` downloads them from the
dataset's GitHub repository
(`https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/`)
into `data/fashion-mnist/` (gitignored) and refuses to read a file whose SHA-256
does not match the hash recorded in `lesson5.FILES`; those hashes were taken
from the files on 2026-10-24 and agree with the MD5 checksums the dataset's
README publishes. CIFAR-10, which the course plan named, was not reachable from
the machine the lesson was written on; Fashion-MNIST has a clear licence, is
small, and has the kind of confusable classes a confusion matrix is for.
