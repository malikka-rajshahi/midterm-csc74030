CV Midterm — Updated TODO List

For the midterm, let’s focus on Shape + Color + Intensity, without additional geometric features.

Once you finish a task, please check it off and add your name next to it so we can easily 
track our progress and keep a record of everyone’s contributions for the midterm presentation.

1. Dataset & Preprocessing

* [ ]	Collect images of orange traffic cones and construction signs
* [ ]	Annotate objects and prepare positive/negative samples
* [ ]	Prepare candidate regions / sliding windows

2. Feature Extraction

* [ ]	Intensity Histogram (Baseline)
* [ ]	HSV Color Histogram
* [ ]	Lab Color Histogram
* [ ]	HOG (Shape Features)
* [ ]	PHOG (Shape Features)

3. Detection Pipeline

* [ ]	Implement feature matching / similarity comparison
* [ ]	Compare individual features (Intensity vs. Color vs. Shape)
* [ ]	Combine Shape + Color + Intensity features
* [ ]	Generate detection bounding boxes and apply NMS

4. Evaluation

* [ ]	Compare feature extraction methods and combinations
* [ ]	Calculate Precision, Recall, and F1
* [ ]	Visualize results and analyze errors

5. Midterm Presentation

* [ ]	Explain methodology and pipeline
* [ ]	Present experimental results
* [ ]	Discuss limitations and final project plans
