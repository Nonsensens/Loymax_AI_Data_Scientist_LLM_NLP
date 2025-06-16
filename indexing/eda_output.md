# Exploratory Data Analysis (EDA)

## Dataset Info

```

<class 'pandas.core.frame.DataFrame'>

RangeIndex: 56952 entries, 0 to 56951

Data columns (total 4 columns):

 #   Column          Non-Null Count  Dtype 

---  ------          --------------  ----- 

 0   uid             56952 non-null  int64 

 1   ru_wiki_pageid  56952 non-null  int64 

 2   text            56952 non-null  object

 3   text_length     56952 non-null  int64 

dtypes: int64(3), object(1)

memory usage: 1.7+ MB

```

## Text Length Statistics

|       |   text_length |
|:------|--------------:|
| count |     56952     |
| mean  |       448.517 |
| std   |       415.26  |
| min   |         1     |
| 25%   |       172     |
| 50%   |       343     |
| 75%   |       600     |
| max   |     11010     |

## Null Values

|                |   0 |
|:---------------|----:|
| uid            |   0 |
| ru_wiki_pageid |   0 |
| text           |   0 |
| text_length    |   0 |
