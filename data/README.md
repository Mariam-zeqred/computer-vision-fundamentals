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
