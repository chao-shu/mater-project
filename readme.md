# Features
1. temporal
    ./Temporal_feature/(video_name).pkl
    dict_strcture:
    {
        'face_(frame_number)_temporal_output': np.array feature(512)
        ...
    }
    feature distribution:
    min:-1.727445 max:1.9969403 mean:0.0040133013 std:0.39413705
    after normalization:
    min:-6.5189695 max:7.8444705 mean:-1.12649005e-07 std:1.0000001
2. spatial
    ./S_former_feature/(video_name).pkl
    dict_structure:
    {
        'face_(frame_number)_mid_output': np.array feature(512)
        'face_(frame_number)_end_output': np.array feature(512)
        ...
    }
    feature distribution
    end: min:0.0 max:3.6462755 mean:0.40473697 std:0.40120935
    mid: min:0.0 max:1.671509 mean:0.19081393 std:0.12601994
    after normalization:
    end: min:-3.9112725 max:42.51135 mean:1.4489227e-07 std:1.0000014
    mid: min:-3.6510618 max:11.099409 mean:-2.019577e-07 std:1.0000005
3. pose
    ./dataset/facial_data/(video_number)/(frame_number).pkl
    dict_structure:
    {
        'img', 
        'face', 
        'aus', 
        'emo', '
        frame_label', 
        'topic_label', 
        'euler_angles', 
        'head_pose', 
        'facial_landmarks', 
        'gaze'
    }
    feature distribution
    without normalization: min:-52.984339782336704 max:1041.1463928222656 mean:499.52197937585805 std:181.0833186513581
    with normalization:min:-7.2298180592479255 max:21.914985804128797 mean:-2.908523868921548e-17 std:0.9999999976163472
4: reduced_pose
    ./reduced_pose_feature/(video_name).pkl
    dict_structure
    {
        00001_face:feature 
        00001_label: 1
    }
NaN detected in: ./dataset/facial_data/1657025477939515/0368.pkl
    'aus', 'emo'
NaN detected in: ./dataset/facial_data/16571735778363602/0417.pkl
    'aus', 'emo'
NaN detected in: ./dataset/facial_data/16571735778363602/0537.pkl
    'aus', 'emo'
NaN detected in: ./dataset/facial_data/16571735778363602/0538.pkl
    'aus', 'emo'
NaN detected in: ./dataset/facial_data/16571735778363602/0539.pkl
    'aus', 'emo'
NaN detected in: ./dataset/facial_data/16571735778363602/0540.pkl
    'aus', 'emo'

total frames 15164


PCA 95%
512 temporal feature -> 28 
original dimension of mid: 256 -> saved dimension of mid 22
original dimension of end: 512 -> saved dimension of end 9
969 pose feature -> 2


after normalization
512 temporal feature -> 38
original dimension of mid: 256 -> saved dimension of mid 29
original dimension of end: 512 -> saved dimension of end 17
969 pose feature -> 3
pose mlp: 128 ->83

feature selected(38+29+17+83) 167-> 31
[90, 93, 102, 119, 147, 89, 84, 85, 94, 135, 100, 145, 134, 120, 101, 107, 92, 118, 162, 123, 88, 31, 36, 160, 86, 148, 163, 87, 95, 60, 103]

feature selection by mlp from 167-> 160
Best selected features: [84, 116, 89, 99, 85, 120, 145, 148, 100, 59, 53, 106, 108, 140, 94, 37, 2, 166, 28, 96, 114, 115, 95, 92, 90, 163, 19, 52, 3, 44, 136, 46, 57, 125, 97, 58, 103, 23, 56, 47, 41, 9, 152, 117, 24, 142, 78, 36, 40, 10, 21, 135, 137, 155, 111, 72, 6, 102, 77, 7, 22, 5, 51, 130, 73, 154, 82, 27, 63, 49, 129, 69, 8, 126, 113, 34, 16, 1, 109, 67, 50, 91, 93, 70, 45, 98, 149, 131, 20, 66, 121, 88, 110, 165, 0, 17, 80, 133, 119, 101, 33, 123, 15, 87, 159, 112, 12, 147, 134, 104, 25, 13, 138, 60, 61, 55, 162, 11, 4, 54, 139, 164, 143, 30, 122, 128, 38, 29, 151, 146, 150, 31, 86, 65, 81, 144, 62, 75, 76, 48, 160, 42, 18, 161, 127, 43, 132, 105, 64, 32, 79, 14, 26, 118, 39, 71, 158, 107, 83, 141]

feature selected by mlp with half-half 167->37
[93, 90, 85, 111, 102, 119, 89, 128, 87, 120, 95, 138, 163, 123, 106, 96, 158, 166, 109, 88, 145, 156, 126, 152, 133, 62, 103, 99, 107, 141, 130, 26, 159, 92, 94, 23, 91]

temporal feature
163904405748.pkl 576
163904657409.pkl 533
163965258378.pkl 577
163974758282.pkl 579
164002054426.pkl 576
16400282145.pkl 577
164006412052.pkl 575
164007913498.pkl 574
164008493608.pkl 577
16569562540764549.pkl 657
1657019500197831.pkl 656
1657025477939515.pkl 685
16570365844633188.pkl 656
16571735778363602.pkl 660
16572034690729465.pkl 657
16572216818408213.pkl 650
165745616715997.pkl 530
16580176851586106.pkl 544
1658828085789384.pkl 581  missing face 539
16637850973931577.pkl 658
16637915350525117.pkl 656
16638397654914925.pkl 483
16638490487013648.pkl 660
16639504750710998.pkl 658
16643616205681827.pkl 629

pose feature
163904405748 576
163904657409 533
163965258378 577
163974758282 579
164002054426 576
16400282145 577
164006412052 575
164007913498 574
164008493608 577
16569562540764549 657
1657019500197831 656
1657025477939515 685
16570365844633188 656
16571735778363602 660
16572034690729465 657
16572216818408213 650
165745616715997 530
16580176851586106 544
1658828085789384 582
16637850973931577 658
16637915350525117 656
16638397654914925 483
16638490487013648 660
16639504750710998 658
16643616205681827 629


worst video in leave one out using kmean selected
-------------------------------------16643616205681827--------------------------------------
Confusion Matrix:
 [[ 50  12   1]
 [ 17 192   0]
 [ 49 307   1]]
Class 1 Accuracy: 79.37%
Class 2 Accuracy: 91.87%
Class 3 Accuracy: 0.28%

Classification Report:
              precision    recall  f1-score   support

     Class 1       0.43      0.79      0.56        63
     Class 2       0.38      0.92      0.53       209
     Class 3       0.50      0.00      0.01       357

    accuracy                           0.39       629
   macro avg       0.44      0.57      0.37       629
weighted avg       0.45      0.39      0.24       629

-------------------------------------1657019500197831--------------------------------------
Confusion Matrix:
 [[  1  62   0]
 [  1 259   1]
 [  1 253  78]]
Class 1 Accuracy: 1.59%
Class 2 Accuracy: 99.23%
Class 3 Accuracy: 23.49%

Classification Report:
              precision    recall  f1-score   support

     Class 1       0.33      0.02      0.03        63
     Class 2       0.45      0.99      0.62       261
     Class 3       0.99      0.23      0.38       332

    accuracy                           0.52       656
   macro avg       0.59      0.41      0.34       656
weighted avg       0.71      0.52      0.44       656

-------------------------------------16572216818408213--------------------------------------
Confusion Matrix:
 [[259  45  34]
 [ 62 130   0]
 [  2 118   0]]
Class 1 Accuracy: 76.63%
Class 2 Accuracy: 67.71%
Class 3 Accuracy: 0.00%

Classification Report:
              precision    recall  f1-score   support

     Class 1       0.80      0.77      0.78       338
     Class 2       0.44      0.68      0.54       192
     Class 3       0.00      0.00      0.00       120

    accuracy                           0.60       650
   macro avg       0.42      0.48      0.44       650
weighted avg       0.55      0.60      0.57       650