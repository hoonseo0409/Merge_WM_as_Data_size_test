# Merge_WM_as_Data_size_test
This code tests one way to solve the Catastrophic Forgetting Problem. In other words, this code was written to compare accuracy of the weight matrix which internally divided between two matrixes as their size of data set.
For example, given weight matrix A, which is the result of learning 75 data, and weight matrix B, which is the result of learning 25 data (not overlapping with the previous 75 data), I tried to guess weight matrix C, which should be the result of learning whole 100 data as like:

C ~ (75 * A + 25 * B) / 100

This code learns from numerical features of the pima-indians-diabetes dataset to predict whether or not it has had diabetes. Model annotated as 10d learns whole data set, model 9d learns dataset from 0% to 90%, and model 1d learns dataset from 90% to 100%.
Because the weight matrix is initialized ramdomly, you will see different results each time you run it, but you will see the following output approximately.

1d result: acc: 61.72%<br/>
10d result: acc: 78.26%<br/>
9d result: acc: 73.96%<br/>
no learn result: acc: 65.10%<br/>
internally divided 9 to 1 result: acc: 69.53%<br/>

The weight matrixes are stored in the /result directory, so if you have an HDF5 viewer, you can open it. The above results suggest that my assumption is wrong. Therefore, we can not solve the catastrophic forgetting problem by simply calculating the center of weight matrixes.
