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
