first develop a frequency table for every word in the training data

uses naive bayes to classify the test data
after stemming
`` train_accuracy = 82.31%``
`` test_accuracy = 70.21% ``
before stemming
``train_accuracy = 85.05%``
``test_accuracy = 66.81%``

random assign: `` 32.78% both train and test accuracies ``
positive assign: `` 43.85% both train and test accuracies``
observation: ``stemming increases test accuracy but decreases train accuracy``

```
part e i
```
calculate the bigrams
retrain the model with the bigrams on stemmed data
``train_accuracy = 94.75%  99.21% with trigrams ``
``test_accuracy = 69.12% 64.96% with trigrams ``


observation: `` using bigrams decreases test accuracy;  increases training accuracy ``

```
part e ii
```
add trigram to the model  
retrain the model with trigram

compare the results of the different models


```
part f
```



with source domain:

1% - `` 46.288% ``
2% - `` 46.322% ``
5% - `` 47.117%``
10% - `` 47.813% ``
25% - `` 44.035% ``
50% - `` 46.090% ``
100% - `` 46.388% ``

without source domain:

1% - `` 34.393% ``
2% - `` 38.104% ``
5% - `` 43.870% ``
10% - `` 47.117% ``
25% - `` 48.376% ``
50% - `` 51.027% ``
100% - `` 55.467% ``



`` SVMs using CVXOPT ``

`` Linear Kernel``
train_accuracy  = 97.37%
val_accuracy    = 92%
support vectors = 670 (14.076%)
b = -3.95075181
time  = 1m57s
`` Gaussian Kernel``
train_accuracy  = 95.02%
val_accuracy    = 93.75%
support vectors = 1092 (22.94%)
b = -1.55535714
time  = 2m3s


`` SVMs using LIBSVM ``

`` Linear Kernel ``
train_accuracy  = 95.38%
test_accuracy   = 94.0%
support_vectors = 1038 (21.79%)
b = -2.7138097483069137
time = 5.4s

||W_cvx - W_lib|| = 404.4433202367066

`` Gaussian Kernel ``
train_accuracy  = 95.13%
test_accuracy   = 93.75%
support_vectors = 1086 (22.84%)
b=-2.597234343350003
time = 5.8s

` if images are resized to 32x32x3 then both training and test accuracies are  `
    `in linear case  (95.37815126050421) and gaussian case (96.26050420168067) `

` Multi class Image Classification `

Kernel = Gaussian(RBF) gamma = 0.001 and C = 1

done ['1', '2'] 1 135.1836154460907
done ['1', '3'] 2 131.8402819633484
done ['1', '4'] 3 114.09448575973511
done ['1', '5'] 4 133.39053058624268
done ['1', '6'] 5 160.38148474693298
done ['2', '3'] 6 176.69651675224304
done ['2', '4'] 7 170.9015600681305
done ['2', '5'] 8 168.28006029129028
done ['2', '6'] 9 119.66135716438293
done ['3', '4'] 10 144.13616943359375
done ['3', '5'] 11 121.76456570625305
done ['3', '6'] 12 132.22760915756226
done ['4', '5'] 13 89.84606599807739
done ['4', '6'] 14 122.0499918460846
done ['5', '6'] 15 101.0111973285675


CVXOPT SVMs
`` Test Accuracy ~ 53.91% ``
Training time = ` 39m 57s `
prediction time = `26m 30s `
[
    [ 76,  15,  22,  29,  21,  37],
    [ 10, 145,   1,   5,   7,  32],
    [ 12,   2, 121,  29,  23,  13],
    [ 31,   5,  24, 124,   9,   7],
    [ 31,  12,  55,  39,  52,  11],
    [ 23,  22,  10,  10,   6, 129]
]
classes 0 and 1 mostly missclassified as 5, 2 as 3,3 as 0,4 as 2,5 as 0 


LIBSVM SVMs
training time = ` 2m 11s `
`` Train Accuracy = 56.85% ``
`` Test Accuracy = 55.91% ``
prediction time = ` 2m 14s `

[
    [ 76,  22,  18,  26,  24,  34],
    [  4, 150,   1,   6,  12,  27],
    [ 11,   4, 125,  26,  22,  12],
    [ 24,   6,  25, 126,  14,   5],
    [ 18,  17,  58,  33,  68,   6],
    [ 25,  23,  10,   8,   8, 126]
]

lesser time and little bit more accuracy than CVXOPT SVMs



            Validation Accuracy     K-fold accuracy

C = 1e-5    40.166666666666664      15.644257703081232
C = 1e-3    40.166666666666664      16.64565826330532
C = 1       55.91666666666667       49.71
C = 5       59.25                   58.58
C = 10      60.83333333333333       63.87