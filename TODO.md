# TODO

## Open (QC 2026-10-13)

- [x] `README.md` is 58 lines; STYLE.md asks for 80–200 with an explicit "Next steps" section. Add "Next steps" (lesson 2 scope, notebook policy) and expand "What the outputs look like" with the printed numbers, without padding. *(2026-10-15: README is now 70+ lines with "Next steps"; lesson rows will keep growing it.)*
- [ ] Lesson 1 has one Mermaid diagram (section 3, colour-space map). The 2026-10-12 LOG.md entry says two. Either add a second (the worked-example pipeline: image → HSV → hue histogram → valley → mask → patch check) or correct the LOG entry.
- [ ] The only commit before this QC is authored `Claude <noreply@anthropic.com>` (the sandbox's git identity). Decide whether future commits should carry Mariam's name/email (set `git config user.*` in the sandbox); do not rewrite history.
- [ ] Mariam: confirm the attribution in `data/README.md` (Rachel Michetti, Pikolo Espresso Bar, CC0) against the scikit-image `skimage.data.coffee` docstring. The sandbox cannot reach scikit-image.org or the scikit-image GitHub repo, so it was not checked externally.
- [x] Lesson 2 (filtering, edges, morphology) should open by cleaning the lesson 1 saucer mask, as WALKTHROUGH.md promises. *(2026-10-15: it does, section 1.)*

## Open (lesson 2, 2026-10-15)

- [ ] Lesson 2's egg counter has only been scored on the synthetic belt. If a real belt or shed photo with a hand count becomes available, add it to `data/` with its licence and report the count honestly, even if it is wrong.
- [ ] The 60 % distance-transform rule is known to fail for uneven egg sizes or three eggs in a row (lesson 2, "What is fragile here"). A `cv2.watershed` version seeded from the peaks is the natural follow-up; WALKTHROUGH.md describes it.
- [ ] Lesson 2 is 3 233 words by `wc -w` (2 413 before the exercises); excluding code blocks, tables and image lines it is 2 847 (2 027 before the exercises). The cap is 3 000. Trim ~250 words if the course's word-count rule counts everything `wc -w` counts, nothing if it counts prose only.
- [ ] `WALKTHROUGH.md` is ~960 words with two lessons; STYLE.md's 300–600 guideline was written for project repos. Decide whether a course walkthrough may grow per lesson (keep as is) or whether each lesson's notes move to `notes/lessonN.md` with WALKTHROUGH.md holding only the course-level part.

## Open (lesson 3, 2026-10-18)

- [ ] Lesson 3 is 3 577 words by `wc -w` (2 997 prose-only, excluding code blocks, tables and image lines; 2 148 before the exercises). Same policy question as lesson 2.
- [ ] The belt has no egg height, so the off-plane (parallax) effect the lesson describes in "What is fragile here" and exercise 3 is stated, not measured. A `make_belt(height_mm=...)` that renders eggs as ellipsoids seen from the camera, or a real oblique photo of eggs with markers, would let the lesson quote a number.
- [ ] ORB finds only 52 ratio matches / 31 inliers on the smooth synthetic belt. That is reported honestly; if a real belt photo pair arrives, re-run `travel_from_matches` and update the numbers either way.
- [ ] `test_harris_matches_opencv_up_to_scale` compares the top-5 % corner sets of `harris_response` (Gaussian window) and `cv2.cornerHarris` (box window) and asks for 50 % overlap; it is a sanity check, not a pin. If OpenCV changes `cornerHarris` normalisation the threshold may need revisiting.
- [ ] Exercise 5's tracker (`track_eggs`) is described in WALKTHROUGH.md and not implemented; it is the natural lesson-3 follow-up and does not need new data.
