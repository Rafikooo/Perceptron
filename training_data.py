import numpy as np

data_0 = np.array([
    [
     1, 1, 1, 1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, 1, 1, 1, 1
    ],
    [
     1, 1, 1, 1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, 1, -1, -1, 1,
     1, 1, 1, 1, 1
    ],
    [
     1, 1, 1, 1, 1,
     1, -1, -1, 1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, 1, 1, 1, 1
    ],
    [
     -1, -1, -1, -1, -1,
     1, 1, 1, 1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, 1, 1, 1, 1
    ],
    [
     1, 1, 1, 1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, 1, 1,
     1, 1, 1, 1, 1
    ]
])

data_1 = np.array([
    [
        -1, -1, -1, -1, 1,
        -1, -1, -1, -1, 1,
        -1, -1, -1, -1, 1,
        -1, -1, -1, -1, 1,
        -1, -1, -1, -1, 1,
        -1, -1, -1, -1, 1,
        -1, -1, -1, -1, 1,
    ],
    [
        -1, -1, -1, 1, -1,
        -1, -1, -1, 1, -1,
        -1, -1, -1, 1, -1,
        -1, -1, -1, 1, -1,
        -1, -1, -1, 1, -1,
        -1, -1, -1, 1, -1,
        -1, -1, -1, 1, -1,
    ],
    [
        -1, -1, 1, -1, -1,
        -1, -1, 1, -1, -1,
        -1, -1, 1, -1, -1,
        -1, -1, 1, -1, -1,
        -1, -1, 1, -1, -1,
        -1, -1, 1, -1, -1,
        -1, -1, 1, -1, -1,
    ],
    [
        -1, 1, -1, -1, -1,
        -1, 1, -1, -1, -1,
        -1, 1, -1, -1, -1,
        -1, 1, -1, -1, -1,
        -1, 1, -1, -1, -1,
        -1, 1, -1, -1, -1,
        -1, 1, -1, -1, -1,
    ],
    [
        -1, -1, -1, -1, 1,
        -1, -1, -1, 1, 1,
        -1, -1, -1, -1, 1,
        -1, -1, -1, -1, 1,
        -1, -1, -1, -1, 1,
        -1, -1, -1, -1, 1,
        -1, -1, -1, -1, 1
    ]
])

data_2 = np.array([
 [
  1, 1, 1, 1, -1,
  1, -1, -1, -1, 1,
  -1, -1, -1, 1, -1,
  -1, -1, 1, -1, -1,
  -1, 1, -1, -1, -1,
  1, -1, -1, -1, 1,
  -1, 1, 1, 1, -1
 ],
 [
  1, 1, 1, 1, 1,
  1, -1, -1, -1, 1,
  -1, -1, -1, 1, -1,
  -1, -1, 1, -1, -1,
  -1, 1, -1, -1, -1,
  1, -1, -1, -1, 1,
  -1, 1, 1, 1, 1
 ],
 [
  1, 1, 1, 1, 1,
  -1, -1, -1, -1, 1,
  -1, -1, -1, 1, -1,
  -1, -1, 1, -1, -1,
  -1, 1, -1, -1, -1,
  1, -1, -1, -1, -1,
  1, 1, 1, 1, 1
 ],
 [
  1, 1, 1, 1, 1,
  1, -1, -1, -1, 1,
  1, -1, -1, 1, -1,
  -1, -1, 1, -1, -1,
  -1, 1, -1, -1, 1,
  1, -1, -1, -1, 1,
  1, 1, 1, 1, 1
 ], [
  1, 1, 1, 1, -1,
  1, 1, -1, -1, 1,
  1, -1, -1, 1, -1,
  -1, -1, 1, -1, -1,
  -1, 1, -1, -1, 1,
  1, -1, -1, 1, 1,
  -1, 1, 1, 1, 1
 ],
])

data_3 = np.array([
 [
  1, 1, 1, 1, 1,
  -1, -1, -1, -1, 1,
  -1, -1, -1, -1, 1,
  1, 1, 1, 1, 1,
  -1, -1, -1, -1, 1,
  -1, -1, -1, -1, 1,
  1, 1, 1, 1, 1,
 ],
 [
  1, 1, 1, 1, 1,
  1, -1, -1, -1, 1,
  -1, -1, -1, -1, 1,
  1, 1, 1, 1, 1,
  -1, -1, -1, -1, 1,
  1, -1, -1, -1, 1,
  1, 1, 1, 1, 1,
 ],
 [
  1, 1, 1, 1, 1,
  1, -1, -1, 1, 1,
  -1, -1, -1, -1, 1,
  1, 1, 1, 1, 1,
  -1, -1, -1, -1, 1,
  1, -1, -1, -1, 1,
  1, 1, 1, 1, 1,
 ],
 [
  1, 1, 1, 1, 1,
  -1, -1, -1, -1, 1,
  1, -1, -1, -1, 1,
  1, 1, 1, 1, 1,
  1, -1, -1, -1, 1,
  -1, -1, -1, -1, 1,
  1, 1, 1, 1, 1,
 ],
 [
  1, 1, 1, 1, -1,
  -1, -1, -1, 1, 1,
  1, -1, -1, -1, 1,
  1, 1, 1, 1, 1,
  1, -1, -1, -1, 1,
  -1, -1, -1, 1, 1,
  1, 1, 1, 1, -1,
 ]
])

data_4 = np.array([
    [1, -1, -1, -1, -1,
     1, -1, -1, -1, -1,
     1, -1, -1, -1, -1,
     1, 1, 1, 1, 1,
     -1, -1, -1, -1, 1,
     -1, -1, -1, -1, 1,
     -1, -1, -1, -1, 1],
    [1, -1, -1, -1, -1,
     1, -1, -1, -1, -1,
     1, -1, -1, -1, -1,
     1, -1, -1, -1, -1,
     1, 1, 1, 1, 1,
     -1, -1, -1, -1, 1,
     -1, -1, -1, -1, 1],
    [1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, 1, 1, 1, 1,
     -1, -1, -1, -1, 1,
     -1, -1, -1, -1, 1,
     -1, -1, -1, -1, 1],
    [1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, 1, 1, 1, 1,
     -1, -1, -1, 1, 1,
     -1, -1, -1, -1, 1,
     -1, -1, -1, -1, 1],
    [1, -1, -1, -1, -1,
     1, -1, -1, -1, -1,
     1, 1, -1, -1, 1,
     -1, 1, 1, 1, 1,
     -1, -1, -1, 1, 1,
     1, -1, -1, -1, 1,
     -1, -1, -1, -1, 1],
])
data_5 = np.array([
    [1, 1, 1, 1, 1,
     1, -1, -1, -1, -1,
     1, -1, -1, -1, -1,
     1, 1, 1, 1, 1,
     -1, -1, -1, -1, 1,
     -1, -1, -1, -1, 1,
     1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, -1,
     1, 1, 1, 1, 1,
     -1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1,
     1, -1, -1, -1, -1,
     1, -1, -1, -1, -1,
     1, 1, 1, 1, 1,
     -1, -1, -1, -1, 1,
     -1, -1, -1, -1, 1,
     -1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, -1,
     1, 1, 1, 1, 1,
     -1, -1, -1, -1, 1,
     -1, -1, -1, 1, 1,
     1, 1, 1, 1, -1],
    [1, 1, 1, 1, 1,
     1, -1, -1, -1, -1,
     1, 1, -1, -1, -1,
     -1, 1, 1, 1, 1,
     -1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, 1, 1, 1, 1],
])

data_6 = np.array([
 [
  1, -1, -1, -1, -1,
  1, -1, -1, -1, -1,
  1, -1, -1, -1, -1,
  1, 1, 1, 1, 1,
  1, -1, -1, -1, 1,
  1, -1, -1, -1, 1,
  1, 1, 1, 1, 1,
 ],
 [
  1, -1, -1, -1, -1,
  1, -1, -1, -1, -1,
  1, -1, -1, -1, -1,
  1, 1, 1, 1, 1,
  1, -1, -1, -1, 1,
  1, 1, -1, 1, 1,
  1, 1, 1, 1, 1,
 ],
 [
  1, 1, -1, -1, -1,
  1, -1, -1, -1, -1,
  1, -1, -1, -1, -1,
  1, 1, 1, 1, 1,
  1, -1, -1, -1, 1,
  1, -1, -1, -1, 1,
  1, 1, 1, 1, 1,
 ],
 [
  1, 1, 1, 1, -1,
  1, -1, -1, -1, -1,
  1, -1, -1, -1, -1,
  1, 1, 1, 1, 1,
  1, -1, -1, -1, 1,
  1, -1, -1, -1, 1,
  1, 1, 1, 1, 1,
 ],
 [
  1, 1, 1, 1, 1,
  1, -1, -1, -1, 1,
  1, -1, -1, -1, -1,
  1, 1, 1, 1, 1,
  1, -1, -1, -1, 1,
  1, -1, -1, -1, 1,
  1, 1, 1, 1, 1,
 ],
])

data_7 = np.array([
    [
        1, 1, 1, 1, 1,
        -1, -1, -1, -1, 1,
        -1, -1, -1, 1, -1,
        -1, -1, 1, -1, -1,
        -1, 1, -1, -1, -1,
        1, -1, -1, -1, -1,
        1, -1, -1, -1, -1,
    ],
    [
        1, 1, 1, 1, 1,
        1, -1, -1, -1, 1,
        -1, -1, -1, 1, -1,
        -1, -1, 1, -1, -1,
        -1, 1, -1, -1, -1,
        1, -1, -1, -1, -1,
        1, -1, -1, -1, -1,
    ],    [
        1, 1, 1, 1, 1,
        1, -1, -1, -1, 1,
        -1, -1, -1, 1, -1,
        -1, -1, 1, -1, -1,
        -1, -1, 1, -1, -1,
        -1, 1, -1, -1, -1,
        1, -1, -1, -1, -1,
    ],
    [
        1, 1, 1, 1, 1,
        1, -1, -1, -1, 1,
        1, -1, -1, 1, -1,
        -1, -1, 1, -1, -1,
        -1, -1, 1, -1, -1,
        -1, 1, -1, -1, -1,
        1, -1, -1, -1, -1,
    ],
    [
        1, 1, 1, 1, 1,
        1, 1, -1, -1, 1,
        1, -1, -1, 1, -1,
        -1, -1, 1, 1, -1,
        -1, -1, 1, -1, -1,
        1, 1, -1, -1, -1,
        1, -1, -1, -1, -1,
    ],
])

data_8 = np.array([
    [
        1, 1, 1, 1, 1,
        1, -1, -1, -1, 1,
        1, -1, -1, -1, 1,
        1, 1, 1, 1, 1,
        1, -1, -1, -1, 1,
        1, -1, -1, -1, 1,
        1, 1, 1, 1, 1
    ],
    [
        1, 1, 1, 1, 1,
        1, 1, -1, 1, 1,
        1, -1, -1, -1, 1,
        1, 1, 1, 1, 1,
        1, -1, -1, -1, 1,
        1, 1, -1, 1, 1,
        1, 1, 1, 1, 1
    ],
    [
        -1, 1, 1, 1, -1,
        1, 1, -1, 1, 1,
        1, -1, -1, -1, 1,
        1, 1, 1, 1, 1,
        1, -1, -1, -1, 1,
        1, 1, -1, 1, 1,
        -1, 1, 1, 1, -1
    ],
    [
        1, 1, 1, 1, 1,
        1, -1, -1, -1, 1,
        1, -1, -1, -1, 1,
        1, -1, -1, -1, 1,
        1, 1, 1, 1, 1,
        1, -1, -1, -1, 1,
        1, 1, 1, 1, 1
    ],
    [
        1, 1, 1, 1, 1,
        1, -1, -1, -1, 1,
        1, 1, 1, 1, 1,
        1, -1, -1, -1, 1,
        1, -1, -1, -1, 1,
        1, -1, -1, -1, 1,
        1, 1, 1, 1, 1
    ],
])

data_9 = np.array([
    [
        1, 1, 1, 1, 1,
        1, -1, -1, -1, 1,
        1, -1, -1, -1, 1,
        1, 1, 1, 1, 1,
        -1, -1, -1, -1, 1,
        -1, -1, -1, -1, 1,
        1, 1, 1, 1, 1
    ],
    [
        1, 1, 1, 1, 1,
        1, -1, -1, -1, 1,
        1, -1, -1, -1, 1,
        1, 1, 1, 1, 1,
        -1, -1, -1, -1, 1,
        1, -1, -1, -1, 1,
        1, 1, 1, 1, 1
    ],
    [
        1, 1, 1, 1, -1,
        1, -1, -1, 1, 1,
        1, -1, -1, -1, 1,
        1, 1, 1, 1, 1,
        -1, -1, -1, 1, 1,
        1, -1, -1, -1, 1,
        1, 1, 1, 1, 1
    ],
    [
        1, 1, 1, 1, 1,
        1, -1, -1, -1, 1,
        1, -1, -1, -1, 1,
        1, 1, 1, 1, 1,
        -1, -1, -1, 1, 1,
        -1, -1, -1, -1, 1,
        -1, 1, 1, 1, 1
    ],
    [
        -1, 1, 1, 1, -1,
        1, -1, -1, 1, 1,
        1, -1, -1, -1, 1,
        -1, 1, 1, 1, 1,
        -1, -1, -1, -1, 1,
        -1, -1, -1, -1, 1,
        1, 1, 1, 1, 1
    ],
])

data = np.concatenate([data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9])

labels = [
 0, 0, 0, 0, 0,
 1, 1, 1, 1, 1,
 2, 2, 2, 2, 2,
 3, 3, 3, 3, 3,
 4, 4, 4, 4, 4,
 5, 5, 5, 5, 5,
 6, 6, 6, 6, 6,
 7, 7, 7, 7, 7,
 8, 8, 8, 8, 8,
 9, 9, 9, 9, 9
]
