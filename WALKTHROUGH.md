# Walkthrough (for Mariam)

This file is for you, not for readers of the course. It grows by one short section per lesson.

## Why the course is shaped this way

The audience is "can write Python, has never made a computer see anything". Six lessons go from arrays to a fine-tuned detector, and each one produces something the next one uses: lesson 1's saucer mask is cleaned up by lesson 2's morphology, lesson 2's egg counter is what lesson 6's detector replaces. The rule that keeps the course honest is that every number in the prose is printed by `lessonN.py` and most are pinned by a test, so if OpenCV changes a formula the test breaks before a reader finds the error.

## File map

- `lessonN-<slug>.md` — one lesson. 1 500–3 000 words, one worked example, exercises with collapsed answers, "What to read next".
- `lessonN.py` — every piece of code the lesson shows, runnable, importable by the tests. Writes `figures/lessonN/`.
- `tests/test_lessonN.py` — pins the numbers the lesson quotes.
- `data/` — images, each with its source and licence in `data/README.md`.
- `README.md` — table of contents; a lesson's row is filled in when it lands.

## Lesson 1 - Images as arrays, colour spaces, histograms

*What it teaches.* Shape is (rows, columns, channels) and the first index is y; `uint8` wraps, so widen before arithmetic; grayscale is a weighted sum (BT.601), not a mean; HSV separates colour from brightness and OpenCV stores hue as 0–179; a histogram is `np.bincount` and its valleys are thresholds; a colour rule inherits the lighting it was tuned under.

*The example.* Separate the red saucer from the brown table in `data/coffee.png`. The red channel cannot do it (saucer 47–119, wood 37–215, overlapping), hue can (saucer 4–6, wood 10–14, valley at 8). The mask also catches the lower half of the white cup, because it is lit by the saucer's reflection; the lesson keeps that in rather than hiding it, because "a white object is not white" is the real lesson.

*Questions a professor may ask.*
1. *Why HSV and not Lab for the threshold?* Because the two populations differ in hue, and hue is one number with a histogram you can read by eye. Lab would separate them too (on the a axis) but needs a 2-D decision. Say that for colour *distance* you would use Lab, for a colour *gate* HSV, and that you know hue is undefined for grey pixels, which is why the saturation guard is there.
2. *Isn't a hand-picked threshold fragile?* Yes, and the lesson says so twice: the threshold comes from this image's histogram, and the mask includes the reflected cup. The honest answer is that `hue_valley()` re-derives the threshold per image, which handles exposure changes but not a different coloured table, and that lesson 5 is where a learned classifier takes over.
3. *Why does the wrap-around matter?* Red straddles hue 0/179. Exercise 3 shows the saucer's hue standard deviation reading 15–18 because of a few pixels at 177–179 when the true spread is 2.3 and 1.3 (both printed by `lesson1.py`). Be ready to explain circular statistics in one sentence (average the unit vectors, not the angles).

*One thing you could extend.* Add a `--valley-only` flag to `lesson1.py` that prints the valley for any image, then run it over a folder of poultry-house frames taken at different times of day and plot how the valley moves; that is a small, real experiment about lighting robustness that no textbook gives you.
