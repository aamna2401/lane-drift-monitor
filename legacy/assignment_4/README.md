# Original assignment (CV course, Assignment 4: Lane Departure System)

Kept for provenance. `code_msds20018_04.py` is the submitted code; `assignment_brief.pdf` is the brief.

What this code did: read an image, crop a triangular ROI, convert to HSV, blur, mask yellow/white with per-pixel Python
loops, run Canny, and stop before the Hough step (Hough, slope grouping and line fitting were commented out).

Known problems, all fixed in `lanedrift/`:
- Functions read globals (`img`, `rgb_img`, `binary_img`), so they cannot be called independently or tested.
- Per-pixel loops (about 100x slower than `cv2.inRange`) and `plt.show()` inside every step.
- The Canny output was inverted into the "binary image", and the ROI was applied before colour conversion.
- Steps 6-8 (Hough, left/right grouping, regression) were not finished; ROI was fixed, not automatic.
- Single images only: no video, no temporal consistency, no metres, no departure logic.
