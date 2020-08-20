# EquSolve
Hand written equation solver

## Intro
This project is an application for scanning an hand written equations from a hardcopy paper. There are 2 main steps:

1. Recognize the equation
2. Solve it using symbolic solver


The first mission consists of detection of symbols and numbers and combining them in the right way. 

The detection could be addressed by using relatively simple computer vision techniques to seperate the symbols and then use a simple machine learning model to recognize which symbol are there. Another approche would be to directly train convolutional neural network of a suitable architecture for object detection (we can use detectron or YOLO for that).

## Benchmarks
There are 3 levels we hope to get to:

### Basic operations:
1. recognizing digits
2. combining digits to numbers
3. recognizing basic operations signs (-+/x*^ sqrt)
4. computing basic operations consisting of numbers and basic operations
### Algebraic equations:
1. recognizing letters as variables
2. solving single variable equation
3. solving multiple variables equations
4. recognizing basic functions (trigo, log) and solving such equations
### Differential equations:
1. recognizing differential operators (integrals, differentials)
2. compute differential operations
3. solving differential equations!
