Used for a midterm project in Enviornmental Planning (CRP2340). Using Google’s Street View Static API and OpenCV, this code analyzes the green view index in Bukahyeon-dong, Seoul, South Korea. The code first collects all street view images every 0.0005° (about 55m). Then by setting the upper and lower limits of ‘green’ in HSV, regarding each pixels’ color - it analyzes the ‘green view index’ in every point in the neighborhood. The final result will be given in a form of heatmap.
