
# Train Fasttext language model using Gensim
We use the *6000 Indian Food Recipes* recipe data from here: 
https://www.kaggle.com/kanishk307/6000-indian-food-recipes-dataset


```python
import pandas as pd
import numpy as np
import collections
import gensim 
from gensim.models import word2vec, phrases
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation, strip_numeric,\
                    strip_non_alphanum, strip_multiple_whitespaces, strip_short
from textblob import TextBlob, Word

import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
%matplotlib inline
```


```python
import os
fc=0
for el in os.listdir("."):
    fc+=1
    print(" - ",el)
print("found",fc,"files!")
```

     -  .ipynb_checkpoints
     -  1_trainRecipeEmbeddings.ipynb
     -  2_indexRecipeData.ipynb
     -  2_trainEmbeddings.ipynb
     -  data
     -  models
     -  processed
     -  recipeEmbeddingsBasedSearch.ipynb
    found 8 files!



```python
raw = pd.read_csv(r"C:\Users\arnab\Documents\workspace\food\whatscooking\downloads\IndianFoodDatasetCSV.csv")
df = raw.copy()
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Srno</th>
      <th>RecipeName</th>
      <th>TranslatedRecipeName</th>
      <th>Ingredients</th>
      <th>TranslatedIngredients</th>
      <th>PrepTimeInMins</th>
      <th>CookTimeInMins</th>
      <th>TotalTimeInMins</th>
      <th>Servings</th>
      <th>Cuisine</th>
      <th>Course</th>
      <th>Diet</th>
      <th>Instructions</th>
      <th>TranslatedInstructions</th>
      <th>URL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Masala Karela Recipe</td>
      <td>Masala Karela Recipe</td>
      <td>6 Karela (Bitter Gourd/ Pavakkai) - deseeded,S...</td>
      <td>6 Karela (Bitter Gourd/ Pavakkai) - deseeded,S...</td>
      <td>15</td>
      <td>30</td>
      <td>45</td>
      <td>6</td>
      <td>Indian</td>
      <td>Side Dish</td>
      <td>Diabetic Friendly</td>
      <td>To begin making the Masala Karela Recipe,de-se...</td>
      <td>To begin making the Masala Karela Recipe,de-se...</td>
      <td>https://www.archanaskitchen.com/masala-karela-...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>टमाटर पुलियोगरे रेसिपी - Spicy Tomato Rice (Re...</td>
      <td>Spicy Tomato Rice (Recipe)</td>
      <td>2-1/2 कप चावल - पका ले,3 टमाटर,3 छोटा चमच्च बी...</td>
      <td>2-1 / 2 cups rice - cooked, 3 tomatoes, 3 teas...</td>
      <td>5</td>
      <td>10</td>
      <td>15</td>
      <td>3</td>
      <td>South Indian Recipes</td>
      <td>Main Course</td>
      <td>Vegetarian</td>
      <td>टमाटर पुलियोगरे बनाने के लिए सबसे पहले टमाटर क...</td>
      <td>To make tomato puliogere, first cut the tomato...</td>
      <td>http://www.archanaskitchen.com/spicy-tomato-ri...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Ragi Semiya Upma Recipe - Ragi Millet Vermicel...</td>
      <td>Ragi Semiya Upma Recipe - Ragi Millet Vermicel...</td>
      <td>1-1/2 cups Rice Vermicelli Noodles (Thin),1 On...</td>
      <td>1-1/2 cups Rice Vermicelli Noodles (Thin),1 On...</td>
      <td>20</td>
      <td>30</td>
      <td>50</td>
      <td>4</td>
      <td>South Indian Recipes</td>
      <td>South Indian Breakfast</td>
      <td>High Protein Vegetarian</td>
      <td>To begin making the Ragi Vermicelli Recipe, fi...</td>
      <td>To begin making the Ragi Vermicelli Recipe, fi...</td>
      <td>http://www.archanaskitchen.com/ragi-vermicelli...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Gongura Chicken Curry Recipe - Andhra Style Go...</td>
      <td>Gongura Chicken Curry Recipe - Andhra Style Go...</td>
      <td>500 grams Chicken,2 Onion - chopped,1 Tomato -...</td>
      <td>500 grams Chicken,2 Onion - chopped,1 Tomato -...</td>
      <td>15</td>
      <td>30</td>
      <td>45</td>
      <td>4</td>
      <td>Andhra</td>
      <td>Lunch</td>
      <td>Non Vegeterian</td>
      <td>To begin making Gongura Chicken Curry Recipe f...</td>
      <td>To begin making Gongura Chicken Curry Recipe f...</td>
      <td>http://www.archanaskitchen.com/gongura-chicken...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>आंध्रा स्टाइल आलम पचड़ी रेसिपी - Adrak Chutney ...</td>
      <td>Andhra Style Alam Pachadi Recipe - Adrak Chutn...</td>
      <td>1 बड़ा चमच्च चना दाल,1 बड़ा चमच्च सफ़ेद उरद दाल,2...</td>
      <td>1 tablespoon chana dal, 1 tablespoon white ura...</td>
      <td>10</td>
      <td>20</td>
      <td>30</td>
      <td>4</td>
      <td>Andhra</td>
      <td>South Indian Breakfast</td>
      <td>Vegetarian</td>
      <td>आंध्रा स्टाइल आलम पचड़ी बनाने के लिए सबसे पहले ...</td>
      <td>To make Andhra Style Alam Pachadi, first heat ...</td>
      <td>https://www.archanaskitchen.com/andhra-style-a...</td>
    </tr>
  </tbody>
</table>
</div>



# Basic insights and data processing


```python
#we keep the Translated versions of Ingredients and Instructions (already pre-processed data)
columns_to_drop = ['RecipeName', 'Ingredients', 'PrepTimeInMins' , 'CookTimeInMins',
                   'TotalTimeInMins', 'Instructions', 'Servings', 'Srno']
df_indianRecipes = df.drop(columns = columns_to_drop).dropna()
df_indianRecipes.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TranslatedRecipeName</th>
      <th>TranslatedIngredients</th>
      <th>Cuisine</th>
      <th>Course</th>
      <th>Diet</th>
      <th>TranslatedInstructions</th>
      <th>URL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Masala Karela Recipe</td>
      <td>6 Karela (Bitter Gourd/ Pavakkai) - deseeded,S...</td>
      <td>Indian</td>
      <td>Side Dish</td>
      <td>Diabetic Friendly</td>
      <td>To begin making the Masala Karela Recipe,de-se...</td>
      <td>https://www.archanaskitchen.com/masala-karela-...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Spicy Tomato Rice (Recipe)</td>
      <td>2-1 / 2 cups rice - cooked, 3 tomatoes, 3 teas...</td>
      <td>South Indian Recipes</td>
      <td>Main Course</td>
      <td>Vegetarian</td>
      <td>To make tomato puliogere, first cut the tomato...</td>
      <td>http://www.archanaskitchen.com/spicy-tomato-ri...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ragi Semiya Upma Recipe - Ragi Millet Vermicel...</td>
      <td>1-1/2 cups Rice Vermicelli Noodles (Thin),1 On...</td>
      <td>South Indian Recipes</td>
      <td>South Indian Breakfast</td>
      <td>High Protein Vegetarian</td>
      <td>To begin making the Ragi Vermicelli Recipe, fi...</td>
      <td>http://www.archanaskitchen.com/ragi-vermicelli...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Gongura Chicken Curry Recipe - Andhra Style Go...</td>
      <td>500 grams Chicken,2 Onion - chopped,1 Tomato -...</td>
      <td>Andhra</td>
      <td>Lunch</td>
      <td>Non Vegeterian</td>
      <td>To begin making Gongura Chicken Curry Recipe f...</td>
      <td>http://www.archanaskitchen.com/gongura-chicken...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Andhra Style Alam Pachadi Recipe - Adrak Chutn...</td>
      <td>1 tablespoon chana dal, 1 tablespoon white ura...</td>
      <td>Andhra</td>
      <td>South Indian Breakfast</td>
      <td>Vegetarian</td>
      <td>To make Andhra Style Alam Pachadi, first heat ...</td>
      <td>https://www.archanaskitchen.com/andhra-style-a...</td>
    </tr>
  </tbody>
</table>
</div>




```python
#we'll use the embeddings for classification later
counts_ingr = collections.Counter(df_indianRecipes['Diet'])
most_common = counts_ingr.most_common()
mc = pd.DataFrame(most_common, columns=['word', 'frequency'])
mc.plot(kind='barh', x='word', color='turquoise',figsize=(7,3))
```




    <AxesSubplot:ylabel='word'>




![png](1_trainRecipeEmbeddings_files/1_trainRecipeEmbeddings_6_1.png)



```python
#histogram on cuisines
#df_indianRecipes['Cuisine'].hist(xrot=90,figsize=(15,10))
counts_ingr = collections.Counter(df_indianRecipes['Cuisine'])
most_common = counts_ingr.most_common(25)
mc = pd.DataFrame(most_common, columns=['word', 'frequency'])

#print("\nMost common:\n", mc)
mc.plot(kind='barh', x='word', align='center', color='turquoise',figsize=(12,7))
```




    <AxesSubplot:ylabel='word'>




![png](1_trainRecipeEmbeddings_files/1_trainRecipeEmbeddings_7_1.png)



```python
#some more processing, dropping columns in hindi, copied from a notebook in kaggle
#df_indianRecipes = df_indianRecipes['TranslatedIngredients']
def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True
    
#create boolean mask
mask = df_indianRecipes['TranslatedIngredients'].apply(isEnglish)
df_indianRecipes = df_indianRecipes[mask].dropna()

print("Shape of recipe data:",df_indianRecipes.shape)
```

    Shape of recipe data: (6173, 7)



```python
#more preprocessing on the text fields
df_indianRecipes['TranslatedIngredients'] = df_indianRecipes['TranslatedIngredients'].apply(lambda x: x.lower())

total_ingredients = []
all_receipes_ingredients =  []
for i in range(len(df_indianRecipes)):
    all_ingredients = list()
    #split each recipe into different ingredients
    try:
        ingred = df_indianRecipes.loc[i, "TranslatedIngredients"][1:-1]
    except:
        continue;
      
    for ing in (ingred.split(',')):
        ing = remove_stopwords(ing)
        ing = strip_numeric(ing)
        ing = re.sub(r'\(.*oz.\)|(®)|(.*ed)|(.*ly)|tast|sprig|inch|purpose|flmy|taste|boneless|skinless|chunks|fresh|large|cook drain|green|frozen|ground|tablespoon|teaspoon|cup','',ing).strip()
        ing = strip_short(ing,2)
        ing = strip_multiple_whitespaces(ing)
        ing = strip_punctuation(ing)
        ing = strip_non_alphanum(ing)
        
        #convert plurals to singular e.g. tomatoes --> tomato
        ing = (" ".join(TextBlob(ing).words.singularize()))
        if len(ing)>0:
            all_ingredients.append(ing)
            total_ingredients.append(ing)
    all_receipes_ingredients.append(all_ingredients)
counts_ingr = collections.Counter(total_ingredients)
```


```python
#find the most common ingredients used across all recipes
print ("---- Most Common Ingredients ----")
print (counts_ingr.most_common(25))

print ("\n")
#find the most common ingredients used across all recipes
print ("---- Least Common Ingredients ----")
print (counts_ingr.most_common()[-10:])
print("type counts ingr",type(counts_ingr))

most_common = counts_ingr.most_common(25)
mc = pd.DataFrame(most_common, columns=['word', 'frequency'])

#print("\nMost common:\n", mc)
mc.plot(kind='barh', x='word', align='center', color='turquoise')
```

    ---- Most Common Ingredients ----
    [('salt', 4315), ('turmeric powder haldi', 1681), ('chilli powder', 1595), ('jeera', 1131), ('sunflower oil', 1030), ('chilly', 842), ('sugar', 779), ('curry leaf', 775), ('asafoetida hing', 712), ('garam masala powder', 699), ('s', 655), ('coriander powder dhanium', 648), ('black pepper powder', 627), ('lemon juice', 612), ('water', 572), ('ghee', 559), ('cumin powder jeera', 535), ('extra virgin olive oil', 503), ('clove garlic', 466), ('milk', 450), ('cinnamon stick dalchini', 440), ('flmy maida', 428), ('wheat flmy', 402), ('curd dahi yogurt', 386), ('clove laung', 369)]
    
    
    ---- Least Common Ingredients ----
    [('cumin powder jeera cumin powder', 1), ('coconut milk consistency', 1), ('coconut oil regular oil', 1), ('turmeric powder haldi to add boiling veggie', 1), ('tamarind paste little', 1), ('gm parwal peel chop straight', 1), ('tsp watermelon', 1), ('egg egg', 1), ('sunflower oil ghee frying parath', 1), ('rose petal garnishing optional', 1)]
    type counts ingr <class 'collections.Counter'>





    <AxesSubplot:ylabel='word'>




![png](1_trainRecipeEmbeddings_files/1_trainRecipeEmbeddings_10_2.png)



```python
#convert to lower case
df_indianRecipes['TranslatedInstructions'] = df_indianRecipes['TranslatedInstructions'].apply(lambda x: x.lower())

#total_ingredients = []
all_instructions =  []

#len(df_indianRecipes)
for i in range(len(df_indianRecipes)):
    #split each recipe into different ingredients
    try:
        instrs = df_indianRecipes.loc[i, "TranslatedInstructions"][1:-1]
        #print("instrs\n",instrs)
    except:
        continue;      
    for instr in (instrs.split('.')):
        instr = remove_stopwords(instr)
        instr = strip_numeric(instr)
        #instr = re.sub(r'\(.*oz.\)|(®)|(.*ed)|(.*ly)|tast|sprig|inch|purpose|flmy|taste|boneless|skinless|chunks|fresh|large|cook drain|green|frozen|ground|tablespoon|teaspoon|cup','',ing).strip()
        instr = strip_short(instr,2)
        instr = strip_multiple_whitespaces(instr)
        instr = strip_punctuation(instr)
        instr = strip_non_alphanum(instr)
        #convert plurals to singular e.g. tomatoes --> tomato
        instr = (" ".join(TextBlob(instr).words.singularize()))
        if len(instr)>0:
            all_instructions.append(instr)

print("len",len(all_instructions))

#formatting the column in a way the gensim takes it
all_instructions_splitted = [sentx.split() for sentx in all_instructions]
df_indianRecipes.to_pickle("processed/df_indianRecipes.pkl")
```

    len 78073



```python
#record the number of ingredients for each recipe, 
#add cleaned instructions for training recipe embeddings
#add cleaned ingredients back to original dataframe
df_indianRecipes['clean_ingredients'] = pd.Series(all_receipes_ingredients)
df_indianRecipes = df_indianRecipes.dropna()
df_indianRecipes['ingredient_count'] =  df_indianRecipes.apply(lambda row: len(row['clean_ingredients']), axis = 1)
df_indianRecipes['clean_instructions'] = df_indianRecipes['TranslatedInstructions'].apply(lambda x: x.lower())
df_indianRecipes['clean_instructions'].head()
```




    0    to begin making the masala karela recipe,de-se...
    1    to make tomato puliogere, first cut the tomato...
    2    to begin making the ragi vermicelli recipe, fi...
    3    to begin making gongura chicken curry recipe f...
    4    to make andhra style alam pachadi, first heat ...
    Name: clean_instructions, dtype: object




```python
df_indianRecipes.to_pickle("processed/df_indianRecipes.pkl")
```


```python
#function to process the column 'recipe' to the format of list-of-lists
import re
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation, strip_numeric, strip_non_alphanum, strip_multiple_whitespaces, strip_short

def process_recipe(recipe):
    recipeProcessed = []
    for ing in (recipe.split(',')):
        ing = remove_stopwords(ing)
        ing = strip_numeric(ing)
        ing = re.sub(r'\(.*oz.\)|(®)|(.*ed)|(.*ly)|tast|sprig|inch|purpose|flmy|taste|boneless|skinless|chunks|fresh|large|cook drain|green|frozen|ground|tablespoon|teaspoon|recipe|cup','',ing).strip()
        ing = strip_short(ing,2)
        ing = strip_multiple_whitespaces(ing)
        ing = strip_punctuation(ing)
        ing = strip_non_alphanum(ing)
        if ing:
            #print("ing",ing)
            recipeProcessed.append(ing)
    #print("recipeProcessed",len(recipeProcessed),recipeProcessed)
    return [oneInstr.split() for oneInstr in recipeProcessed]

process_recipe(df_indianRecipes['clean_instructions'][10])
```




    [['beans'],
     ['step',
      'cook',
      'them',
      'like',
      'pressure',
      'cooker',
      'method',
      'cook',
      'beans',
      'place',
      'beans',
      'pressure'],
     ['place', 'weight'],
     ['cover', 'pan', 'cook', 'water', 'heat', 'whistles'],
     ['ium',
      'heat',
      'add',
      'onions',
      'garlic',
      'light',
      'sauté',
      'onions',
      'tender',
      'add',
      'tomatoes'],
     ['tomatoes', 'cover', 'pan'],
     ['beans', 'coriander', 'leaves', 'serve', 'toast'],
     ['potato', 'cakes']]




```python
#adding clean instructions
df_indianRecipes['clean_instructions'] = df_indianRecipes['clean_instructions'].apply(lambda x:process_recipe(x))
print(df_indianRecipes['clean_instructions'][1])
```

    [['tomato', 'puliogere'], ['cut', 'tomatoes', 'mixer', 'grinder', 'puree', 'it', 'heat', 'oil', 'pan', 'oil', 'hot'], ['add', 'chana', 'dal'], ['urad', 'dal'], ['cashew', 'let', 'cook', 'seconds', 'seconds'], ['chillies'], ['chillies', 'curry', 'leaves', 'seconds'], ['add', 'tomato', 'puree', 'mix', 'add', 'bc', 'belle', 'bhat', 'powder'], ['salt', 'mix', 'it', 'allow', 'cook', 'minutes', 'turn', 'gas', 'bowl'], ['rice', 'mix', 'it', 'serve', 'hot', 'serve', 'tomato', 'puliogre', 'tomato', 'cucumber', 'raita', 'papad', 'dinner']]



```python
sentenced_Instructions = df_indianRecipes['clean_instructions']
#all_Sentences is the list of sentences in a format that gensim uses
#list of sentences, where every sentence is a list of words
all_Sentences = []
for sentence in sentenced_Instructions:
    all_Sentences.extend(sentence)
print(len(sentenced_Instructions),"recipes!")
print(len(all_Sentences),"sentences!")
all_Sentences
```

    5020 recipes!
    49899 sentences!





    [['begin', 'making', 'masala', 'karela'],
     ['karela',
      'slice',
      'remove',
      'skin',
      'skin',
      'nutrients',
      'add',
      'karela',
      'pressure',
      'cooker',
      'water'],
     ['add',
      'onions',
      'saute',
      'till',
      'turns',
      'golden',
      'brown',
      'color',
      'add',
      'karela'],
     ['chilli', 'powder'],
     ['amchur', 'powder'],
     ['tomato', 'puliogere'],
     ['cut',
      'tomatoes',
      'mixer',
      'grinder',
      'puree',
      'it',
      'heat',
      'oil',
      'pan',
      'oil',
      'hot'],
     ['add', 'chana', 'dal'],
     ['urad', 'dal'],
     ['cashew', 'let', 'cook', 'seconds', 'seconds'],
     ['chillies'],
     ['chillies', 'curry', 'leaves', 'seconds'],
     ['add', 'tomato', 'puree', 'mix', 'add', 'bc', 'belle', 'bhat', 'powder'],
     ['salt', 'mix', 'it', 'allow', 'cook', 'minutes', 'turn', 'gas', 'bowl'],
     ['rice',
      'mix',
      'it',
      'serve',
      'hot',
      'serve',
      'tomato',
      'puliogre',
      'tomato',
      'cucumber',
      'raita',
      'papad',
      'dinner'],
     ['begin', 'making', 'ragi', 'vermicelli'],
     ['firm',
      'keep',
      'aside',
      'aside',
      'till',
      'later',
      'use',
      'add',
      'drops',
      'oil',
      'mix',
      'don',
      't',
      'stick',
      'other',
      'place',
      'kadai',
      'heat'],
     ['urad', 'dal', 'curry', 'leaves'],
     ['then'],
     ['add', 'onions', 'fry', 'till', 'translucent', 'soft', 'next'],
     ['then'],
     ['switch', 'heat'],
     ['vermicelli',
      'serving',
      'dish',
      'lemon',
      'juice',
      'mix',
      'serve',
      'coconut',
      'chutney',
      'hot',
      'coffee',
      'tea',
      'wholesome',
      'breakfast'],
     ['ients', 'aside', 'in', 'small', 'pan'],
     ['ium', 'heat'],
     ['crackling'],
     ['onions'],
     ['ginger'],
     ['tender', 'add', 'tomatoes'],
     ['saute', 'tomatoes', 'soft', 'mushy', 'add', 'chicken'],
     ['garam',
      'masala',
      'turmeric',
      'powder',
      'add',
      'water',
      'pressure',
      'cook',
      'chicken',
      'whistles',
      'turn',
      'heat',
      'once'],
     ['make',
      'sure',
      'stems',
      'pick',
      'gongura',
      'leaves',
      'wash',
      'chop',
      'gongura',
      'leaves',
      'aside',
      'in',
      'pan'],
     ['saute', 'gongura', 'masala', 'minutes', 'like', 'mushy', 'paste', 'once'],
     ['chicken',
      'curry',
      'gongura',
      'masala',
      'saute',
      'high',
      'heat',
      'minutes',
      'turn',
      'heat',
      'check',
      'salt',
      'spices',
      'adjust',
      'according',
      'e',
      'transfer',
      'gongura',
      'chicken',
      'curry',
      'serving',
      'bowl',
      'serve',
      'hot',
      'serve',
      'gongura',
      'chicken',
      'ghee',
      'rice',
      'neychoru'],
     ['tomato',
      'onion',
      'cucumber',
      'raita',
      'semiya',
      'payasam',
      'delicious',
      'weekend',
      'lunch'],
     ['andhra', 'style', 'alam', 'pachadi'],
     ['chillies'],
     ['onion'],
     ['add',
      'tomatoes',
      'cook',
      'till',
      'tomatoes',
      'soft',
      'turn',
      'gas',
      'allow',
      'cool',
      'cools'],
     ['mixer', 'grinder', 'paste', 'tempering'],
     ['let', 'cook', 'seconds', 'add', 'curry', 'leaves'],
     ['asafoetida',
      'let',
      'cook',
      'seconds',
      'add',
      'chutney',
      'mix',
      'serve',
      'andhra',
      'style',
      'alam',
      'pachadi',
      'ghee',
      'roast',
      'dosa',
      'kirai',
      'sambar',
      'breakfast'],
     ['begin', 'making', 'pudina', 'khara', 'pongal'],
     ['wash', 'soak', 'rice', 'dal', 'minutes', 'make', 'paste', 'mint', 'pudina'],
     ['coriander'],
     ['chilli', 'ginger', 'mixer', 'grinder', 'aside', 'now'],
     ['heat', 'oil', 'pressure', 'cooker'],
     ['rice',
      'dal',
      'fry',
      'minutes',
      'add',
      'mint',
      'coriander',
      'paste',
      'saute',
      'minute',
      'add',
      'water'],
     ['weekend', 'breakfast'],
     ['begin',
      'making',
      'udupi',
      'style',
      'ash',
      'gourd',
      'coconut',
      'curry',
      'cook',
      'ash',
      'gourd',
      'pressure',
      'cooker',
      'little',
      'water'],
     ['soak', 'tamarind', 'ball', 'hot', 'water', 'minutes'],
     ['ium', 'heat'],
     ['transfer', 'bowl', 'for', 'seasoning'],
     ['rice'],
     ['elai', 'vadam', 'pisarna', 'manga', 'lunch'],
     ['begin', 'making', 'black', 'bean', 'burrito'],
     ['pick', 'coriander', 'leaves', 'set', 'aside'],
     ['ium', 'heat'],
     ['beans'],
     ['mix',
      'cook',
      'minute',
      'add',
      'rice',
      'cook',
      'minutes',
      'remove',
      'heat',
      'add',
      'coriander',
      'leaves',
      'kept',
      'aside',
      'salsa'],
     ['toss', 'tomatoes'],
     ['onions'],
     ['spring', 'onions'],
     ['lime', 'juice'],
     ['coriander', 'leaves'],
     ['rice', 'beans'],
     ['generous',
      'spoonful',
      'salsa',
      'grate',
      'cheese',
      'add',
      'spoon',
      'yoghurt',
      'wrap',
      'burrito'],
     ['ients',
      'rest',
      'burritos',
      'serve',
      'black',
      'bean',
      'burrito',
      'chips',
      'healthy',
      'apple'],
     ['radish',
      'pepper',
      'salad',
      'glass',
      'carrot',
      'pineapple',
      'orange',
      'juice',
      'delicious',
      'lunch',
      'dinner'],
     ['prepare', 'spicy', 'crunchy', 'masala', 'idli'],
     ['crisp', 'transfer', 'bowl', 'aside', 'in', 'oil', 'pan', 'add', 'onions'],
     ['capsicum'],
     ['tomatoes'],
     ['salt'],
     ['pepper'],
     ['turmeric', 'powder'],
     ['stir', 'tomato', 'ketchup'],
     ['date', 'chutney', 'tomato', 'ketchup'],
     ['cauliflower', 'leaf', 'chutney'],
     ['leaf',
      'inside',
      'cabbage',
      'wash',
      'well',
      'dry',
      'cut',
      'well',
      'heat',
      'oil',
      'pan',
      'add',
      'urad',
      'dal'],
     ['chilli',
      'cook',
      'till',
      'dal',
      'turns',
      'golden',
      'aside',
      'add',
      'oil',
      'pan',
      'add',
      'cabbage',
      'leaves'],
     ['salt', 'cook', 'till', 'leaves', 'soft', 'add', 'tomatoes'],
     ['garlic', 'mix', 'well', 'cook', 'tomatoes', 'soft', 'cooking'],
     ['turn',
      'gas',
      'allow',
      'mixture',
      'cool',
      'pour',
      'mixer',
      'grinder',
      'dal',
      'mixture',
      'grind',
      'it',
      'add',
      'water',
      'salt',
      'grind',
      'more',
      'tempering'],
     ['tempering', 'pan', 'seconds'],
     ['add', 'cumin'],
     ['asafoetida', 'curry', 'leaves', 'seconds'],
     ['add',
      'chutney',
      'mix',
      'serve',
      'cabbage',
      'leaves',
      'chutney',
      'vegetable',
      'sambar',
      'rice',
      'dinner'],
     ['beans'],
     ['step',
      'cook',
      'them',
      'like',
      'pressure',
      'cooker',
      'method',
      'cook',
      'beans',
      'place',
      'beans',
      'pressure'],
     ['place', 'weight'],
     ['cover', 'pan', 'cook', 'water', 'heat', 'whistles'],
     ['ium',
      'heat',
      'add',
      'onions',
      'garlic',
      'light',
      'sauté',
      'onions',
      'tender',
      'add',
      'tomatoes'],
     ['tomatoes', 'cover', 'pan'],
     ['beans', 'coriander', 'leaves', 'serve', 'toast'],
     ['potato', 'cakes'],
     ['begin', 'making', 'veg', 'chili', 'cheese', 'burgers'],
     ['peel', 'skin', 'add', 'bowl', 'mash', 'potato', 'masher', 'add', 'onions'],
     ['cutlets', 'leave', 'fridge', 'rest', 'sauce', 'to', 'sauce'],
     ['ium', 'heat'],
     ['tomatoes',
      'sprinkle',
      'bit',
      'salt',
      'saute',
      'till',
      'turn',
      'mushy',
      'add',
      'tomato',
      'puree',
      'seasonings',
      'like'],
     ['chilli', 'powder'],
     ['cumin', 'powder'],
     ['chilli', 'flakes'],
     ['oregano'],
     ['notice',
      'tomatoes',
      'change',
      'color',
      'raw',
      'smell',
      'goes',
      'away',
      'at',
      'end',
      'add',
      'soya',
      'granules',
      'salt',
      'e',
      'add',
      'water',
      'allow',
      'boil',
      'minutes',
      'minutes',
      'let',
      'sauce',
      'little',
      'gravish',
      'switch',
      'heat'],
     ['patties',
      'hot',
      'skillet',
      'drizzle',
      'oil',
      'top',
      'cook',
      'atleast',
      'minutes',
      'till',
      'brown',
      'color',
      'crispy',
      'layer',
      'top',
      'do',
      'rest',
      'patties',
      'set',
      'aside',
      'once'],
     ['patty'],
     ['spoon', 'sauce'],
     ['place',
      'cheese',
      'slice',
      'microwave',
      'patty',
      'seconds',
      'cheese',
      'melt',
      'take',
      'patty'],
     ['tea', 'complete', 'meal'],
     ['begin',
      'making',
      'andhra',
      'style',
      'ingavu',
      'chaaru',
      'pressure',
      'cook',
      'tamarind',
      'water',
      'toor',
      'dal'],
     ['mash', 'dal', 'bring', 'mixture', 'mixing', 'well', 'temper'],
     ['heat', 'kadai', 'oil'],
     ['allow', 'crackle', 'add', 'curry', 'leaves'],
     ['comforting', 'meal', 'week', 'day'],
     ['begin', 'making', 'aar', 'macher', 'jhol'],
     ['marinate',
      'fish',
      'salt',
      'turmeric',
      'powder',
      'heat',
      'oil',
      'wok',
      'shallow',
      'fry',
      'fish',
      'sides',
      'till',
      'turn',
      'crispy',
      'golden',
      'remove',
      'aside',
      'into',
      'wok'],
     ['tomatoes', 'sauté', 'well', 'bengali', 'style', 'fish', 'gravy'],
     ['add', 'turmeric', 'powder'],
     ['cumin', 'powder'],
     ['chilies',
      'sauté',
      'till',
      'raw',
      'smell',
      'goes',
      'away',
      'oil',
      'floats',
      'now',
      'add',
      'water',
      'allow',
      'fish',
      'gravy',
      'come',
      'boil',
      'gravy',
      'starts',
      'boiling'],
     ['add', 'slit', 'chilies'],
     ['rice', 'begun', 'bhaja', 'weekday', 'lunch', 'dinner', 'enjoy'],
     ['begin', 'saunf', 'aloo'],
     ['heat', 'oil', 'pressure', 'cooker', 'add', 'turmeric', 'powder'],
     ['salt'],
     ['coriander',
      'serve',
      'sauf',
      'aloo',
      'amritsari',
      'dal',
      'wheat',
      'lachha',
      'paratha',
      'weekday',
      'meal'],
     ['south', 'indian', 'onion', 'chutney'],
     ['chillies',
      'let',
      'cook',
      'seconds',
      'add',
      'urad',
      'dal',
      'let',
      'cook',
      'till',
      'golden',
      'turn',
      'gas',
      'drain',
      'bowl',
      'add',
      'spoon',
      'oil',
      'pan',
      'add',
      'onions',
      'let',
      'cook',
      'minutes',
      'turn',
      'gas',
      'let',
      'cool',
      'down',
      'urad',
      'dal'],
     ['onions'],
     ['tamarind', 'jaggery', 'add', 'water', 'grind', 'it', 'tempering'],
     ['curry', 'leaves', 'seconds'],
     ['add',
      'chutney',
      'mix',
      'serve',
      'south',
      'indian',
      'onion',
      'chutney',
      'masala',
      'dosa',
      'ghee',
      'dosa',
      'tsunag',
      'breakfast'],
     ['begin', 'making', 'hariyali', 'egg', 'curry'],
     ['ium', 'jar'],
     ['add', 'coriander', 'leaves', 'mint', 'leaves'],
     ['chillies',
      'water',
      'blend',
      'smooth',
      'paste',
      'aside',
      'into',
      'small',
      'jar',
      'mixer',
      'grinder'],
     ['add', 'onions'],
     ['ium', 'heat'],
     ['add', 'bay', 'leaf', 'tej', 'patta'],
     ['cloves'],
     ['add', 'garam', 'masala', 'powder'],
     ['coriander',
      'powder',
      'saute',
      'seconds',
      'add',
      'hariyali',
      'coriander',
      'mint',
      'mixture'],
     ['adjust',
      'consistency',
      'hariyali',
      'egg',
      'curry',
      'cover',
      'simmer',
      'hariyali',
      'egg',
      'curry',
      'minutes',
      'check',
      'salt',
      'spices',
      'adjust',
      'according',
      'transfer',
      'hariyali',
      'egg',
      'curry',
      'serving',
      'bowl',
      'serve',
      'hariyali',
      'egg',
      'curry',
      'malabar',
      'parotta',
      'wheat',
      'lachha',
      'paratha',
      'burani',
      'raita',
      'weeknight',
      'dinner'],
     ['prepare', 'gourd', 'raita'],
     ['gourd'],
     ['cucumber'],
     ['curd'],
     ['chillies'],
     ['salt'],
     ['cumin',
      'powder',
      'coriander',
      'bowl',
      'mix',
      'raita',
      'ready',
      'serve',
      'gourd',
      'raita',
      'garlic',
      'dal'],
     ['gourd', 'elder', 'phulka', 'dinner'],
     ['begin', 'making', 'homemade', 'tater', 'tots'],
     ['wash',
      'clean',
      'peel',
      'skin',
      'potatoes',
      'parboil',
      'potatoes',
      'hot',
      'water',
      'saucepan'],
     ['grate', 'potatoes', 'mixing', 'bowl', 'add', 'garlic', 'powder'],
     ['onion', 'powder'],
     ['oregano'],
     ['chili', 'flakes'],
     ['coriander', 'leaves'],
     ['salt'],
     ['flour'],
     ['ium', 'heat', 'golden', 'brown', 'sides', 'once'],
     ['remove',
      'tater',
      'tots',
      'oil',
      'place',
      'kitchen',
      'paper',
      'towels',
      'absorb',
      'excess',
      'oil',
      'serve',
      'tater',
      'tots',
      'tomato',
      'ketchup',
      'sichuan',
      'style',
      'bird',
      'eye',
      'chili',
      'sauce',
      'tea',
      'time',
      'snack',
      'appetizer',
      'parties'],
     ['chettinad', 'vegetable', 'casserole'],
     ['cut',
      'vegetables',
      'soak',
      'rice',
      'minutes',
      'heat',
      'pan',
      'gas',
      'add',
      'long'],
     ['cinnamon'],
     ['fennel'],
     ['black', 'pepper'],
     ['roast',
      'minutes',
      'turn',
      'gas',
      'allow',
      'spice',
      'mixture',
      'cool',
      'cools'],
     ['blender', 'grind', 'it', 'aside', 'heat', 'ghee', 'pan', 'add', 'ginger'],
     ['garlic'],
     ['onion'],
     ['chilies', 'cook', 'till', 'onion', 'soft', 'onion', 'soft'],
     ['add', 'carrots'],
     ['beans'],
     ['potatoes'],
     ['bay', 'leaves'],
     ['turmeric', 'powder'],
     ['chilli', 'powder'],
     ['spices', 'mix', 'well', 'minutes'],
     ['add', 'rice'],
     ['salt'],
     ['water', 'let', 'boil', 'boiling'],
     ['uce', 'heat'],
     ['rice', 'remains', 'raw'],
     ['add', 'water', 'cook', 'it', 'cooking'],
     ['turn', 'gas'],
     ['papad', 'dinner'],
     ['garlic', 'amla', 'chutney'],
     ['wash', 'amla'],
     ['s', 'wash', 'mint'],
     ['coriander', 'aside', 'heat', 'oil', 'pan', 'add', 'small', 'onion'],
     ['garlic', 'cook', 'onion', 'soft', 'add', 'amla'],
     ['mint'],
     ['coriander',
      'cook',
      'minutes',
      'turn',
      'gas',
      'allow',
      'mixture',
      'cool',
      'pour',
      'mixture',
      'mixer',
      'grinder',
      'salt'],
     ['tempering'],
     ['chana', 'dal'],
     ['urad', 'dal'],
     ['curry', 'leaves'],
     ['cashews',
      'cook',
      'cashew',
      'light',
      'brown',
      'add',
      'tempering',
      'chutney',
      'mix',
      'mix',
      'serve',
      'serve',
      'garlic',
      'amla',
      'chutney',
      'ghee',
      'roast',
      'dosa',
      'filter',
      'coffee',
      'breakfast'],
     ['begin', 'making', 'maharashtrian', 'kakdi', 'thalipeeth'],
     ['cucumber'],
     ['turmeric', 'powder'],
     ['cumin', 'powder'],
     ['coriander', 'powder'],
     ['chili', 'powder'],
     ['chillies'],
     ['coriander', 'leaves'],
     ['salt'],
     ['ium', 'heat', 'after', 'minutes'],
     ['ium', 'heat', 'minutes'],
     ['ure',
      'portions',
      'kakdi',
      'thalipeeth',
      'dough',
      'serve',
      'maharashtrian',
      'kakdi',
      'thalipeeth',
      'homemade',
      'yogurt',
      'lahsun',
      'chutney',
      'breakfast'],
     ['begin', 'making', 'murungakkai', 'poriyal'],
     ['wash',
      'drumsticks',
      'peel',
      'removing',
      'tough',
      'part',
      'cut',
      'es',
      'length',
      'take',
      'glass',
      'water',
      'kadhi'],
     ['bring', 'boil'],
     ['add', 'drumsticks', 'salt'],
     ['steam',
      'minutes',
      'careful',
      'overcook',
      'drumsticks',
      'tender',
      'mushy',
      'sambar',
      'avial',
      'drain',
      'water'],
     ['grate', 'coconut'],
     ['add', 'chillies', 'grind', 'mixer', 'in', 'kadai'],
     ['add', 'coconut', 'oil'],
     ['hot'],
     ['chillies'],
     ['coconut', 'stir', 'serve', 'murungakkai', 'poriyal', 'kerala', 'avial'],
     ['rice', 'malabar', 'style', 'dates', 'pickle', 'complete', 'meal'],
     ['begin', 'making', 'kesar', 'mango', 'lassi'],
     ['ients', 'well', 'pulp', 'mangoes', 'aside', 'in', 'blender'],
     ['add', 'mango', 'pulp'],
     ['curd', 'yogurt'],
     ['once'],
     ['almonds', 's'],
     ['honey', 'sugar'],
     ['ice',
      'cubes',
      'blend',
      'kesar',
      'mango',
      'lassi',
      'kesar',
      'mango',
      'lassi',
      'ready',
      'serve'],
     ['almonds',
      'saffron',
      'serve',
      'bhel',
      'puri',
      'asian',
      'watermelon',
      'salad',
      'evening'],
     ['chettinad', 'sweet', 'paniyaram'],
     ['mix', 'mix', 'idli', 'dosa', 'mixture'],
     ['rice', 'rava'],
     ['jaggery'],
     ['coconut'],
     ['salt'],
     ['elaichi', 'powder'],
     ['water',
      'mixing',
      'bowl',
      'aside',
      'minutes',
      'mix',
      'again',
      'heat',
      'paniyaram',
      'pan',
      'add',
      'oil',
      'hole',
      'heating'],
     ['flip',
      'cook',
      'side',
      'cook',
      'sides',
      'golden',
      'remove',
      'pan',
      'serve',
      'serve',
      'chettinad',
      'sweet',
      'paniyaram',
      'masala',
      'tea',
      'filter',
      'coffee',
      'evening',
      'snack'],
     ['begin', 'making', 'mini', 'dal', 'samosa', 'gravy'],
     ['pan'],
     ['ium', 'flame', 'oil', 'hot'],
     ['add', 'cardamom'],
     ['black', 'pepper'],
     ['cloves'],
     ['cinnamon'],
     ['javitri', 'saute', 'seconds', 'add', 'cashew', 'nuts'],
     ['chilli'],
     ['ginger'],
     ['onion',
      'tomatoes',
      'mix',
      'add',
      'water',
      'salt',
      'as',
      'e',
      'cover',
      'pan',
      'boil',
      'mixture',
      'onions',
      'tomato',
      'soft',
      'turn',
      'flame',
      'let',
      'mixture',
      'cool',
      'mixture',
      'cool'],
     ['pan'],
     ['heat', 'oil', 'add', 'bay', 'leaf'],
     ['black',
      'cardamom',
      'spices',
      'release',
      'aromas',
      'add',
      'tomato',
      'onion',
      'mixture'],
     ['chilli', 'powder'],
     ['turmeric', 'powder'],
     ['coriander', 'powder'],
     ['kasuri',
      'methi',
      'mix',
      'cook',
      'mins',
      'low',
      'flame',
      'add',
      'cream',
      'sugar',
      'stir',
      'gravy',
      'simmer',
      'minutes',
      'add',
      'mini',
      'samosas',
      'gravy',
      'turn',
      'flame',
      'make',
      'sure',
      'add',
      'mini',
      'dal',
      'samosas',
      'gravy',
      'hour',
      'serving'],
     ['soggy',
      'mini',
      'dal',
      'samosas',
      'home',
      'follow',
      'moong',
      'dal',
      'mini',
      'samosas',
      'serve',
      'mini',
      'dal',
      'samosa',
      'curry',
      'hot',
      'phulka'],
     ['tawa', 'paratha', 'jeera', 'rice', 'kachumber', 'salad', 'cucumber'],
     ['onion', 'tomatoes', 'sumptuous', 'sunday', 'meal'],
     ['bell', 'pepper', 'sauce'],
     ['slice', 'bread'],
     ['garlic', 'butter'],
     ['meanwhile'],
     ['in', 'pan'],
     ['bell', 'peppers'],
     ['onions'],
     ['salt'],
     ['baguette', 'bread'],
     ['tomatoes'],
     ['cherry',
      'tomato',
      'soup',
      'choice',
      'soup',
      'salad',
      'like',
      'potato',
      'cucumber',
      'salad',
      'light',
      'dinner'],
     ['begin', 'making', 'dal', 'pakwan', 'chaat'],
     ['pakwan', 'ready', 'sift', 'flours', 'together', 'add', 'ajwain'],
     ['cumin'],
     ['pakwan', 'pierce', 'sides'],
     ['drain',
      'water',
      'cook',
      'chana',
      'dal',
      'water',
      'pressure',
      'cooker',
      'couple',
      'whistles',
      'salt'],
     ['sugar'],
     ['turmeric', 'powder'],
     ['garam', 'masala', 'amchur', 'powder', 'after', 'whistles'],
     ['in', 'small', 'tadka', 'pan'],
     ['ium', 'heat', 'add', 'cumin', 'chilies', 'saute', 'seconds'],
     ['add',
      'curry',
      'leaves',
      'asafoetida',
      'turn',
      'heat',
      'pour',
      'seasoning',
      'sindhi',
      'chana',
      'dal',
      'stir',
      'to',
      'mini',
      'dal',
      'pakwan',
      'chaatwhen',
      'ready',
      'serve'],
     ['onions', 'pomegranates'],
     ['sev',
      'serve',
      'these',
      'mini',
      'dal',
      'pakwan',
      'chaat',
      'great',
      'appetizers',
      'parties'],
     ['serve',
      'delicious',
      'weekend',
      'breakfast',
      'well',
      'serve',
      'mini',
      'dal',
      'pakwan',
      'chaat',
      'evening',
      'snack',
      'appetizer',
      'parties',
      'pav',
      'bhaji'],
     ['dahi', 'vada'],
     ['end', 'paneer', 'kheer', 'dessert'],
     ['begin', 'making', 'asian', 'style', 'sweet', 'spicy', 'beans'],
     ['remain'],
     ['ium',
      'heat',
      'oil',
      'hot',
      'add',
      'cloves',
      'saute',
      'seconds',
      'aromas',
      'come',
      'through',
      'stage'],
     ['add', 'soy', 'sauce'],
     ['thai', 'sweet', 'chilli', 'sauce'],
     ['rice', 'wine', 'vinegar'],
     ['sweet', 'spicy', 'sauce', 'once'],
     ['turn',
      'heat',
      'transfer',
      'asian',
      'style',
      'sweet',
      'spicy',
      'beans',
      'serving',
      'bowl',
      'serve',
      'hot',
      'serve',
      'asian',
      'style',
      'sweet',
      'spicy',
      'beans',
      'steaming',
      'hot',
      'thai',
      'jasmine',
      'sticky',
      'rice',
      'thai',
      'pineapple',
      'vegetarian',
      'curry',
      'delicious',
      'meal'],
     ['begin', 'making', 'chinese', 'darsaan', 'ice', 'cream'],
     ['flat', 'noodles', 'flat', 'noodles', 'mixing', 'bowl'],
     ['combine', 'flour'],
     ['baking', 'powder'],
     ['oil'],
     ['water'],
     ['smooth', 'soft', 'dough', 'dough', 'ready'],
     ['balls',
      'dust',
      'dough',
      'balls',
      'flour',
      'roll',
      'like',
      'roti',
      'rolling',
      'pin',
      'paring',
      'knife'],
     ['cut', 'long', 'strips', 'dough', 'measuring', 'centimeter', 'size'],
     ['ium', 'flame'],
     ['oil', 'bring', 'boil', 'once', 'water', 'come', 'rolling', 'boil'],
     ['noodle', 'strips', 'in', 'boil', 'minutes', 'noodles'],
     ['tend', 'fluff'],
     ['remove', 'water'],
     ['spoon', 'strainer', 'that', 's', 'resting', 'bowl'],
     ['ium', 'high', 'heat'],
     ['noodles'],
     ['noodles', 'crispy'],
     ['pale',
      'colour',
      'deep',
      'brown',
      'remove',
      'noodles',
      'kadai',
      'place',
      'absorbent',
      'paper',
      'drain',
      'excess',
      'oil',
      'allow',
      'cool',
      'honey',
      'syrup',
      'small',
      'saucepan'],
     ['combine', 'sugar'],
     ['ium', 'high', 'flame', 'turn', 'flame'],
     ['good', 'mix', 'allow', 'honey', 'syrup', 'cool', 'warm'],
     ['this',
      'darsaan',
      'serve',
      'darsaan',
      'scoop',
      'vanilla',
      'ice',
      'cream',
      'serve',
      'chinese',
      'darsaan',
      'ice',
      'cream',
      'dessert',
      'meal',
      'dragon',
      'chicken',
      'sizzler',
      'sweet',
      'sour',
      'vegetable',
      'tofu',
      'brown',
      'rice'],
     ['sundakai', 'methi', 'sambar'],
     ['mix',
      'tamarind',
      'water',
      'bowl',
      'heat',
      'oil',
      'pan',
      'add',
      'asafetida',
      'cook',
      'seconds',
      'seconds'],
     ['crackle',
      'add',
      'sundkai',
      'cook',
      'minutes',
      'add',
      'curry',
      'leaves',
      'mix',
      'add',
      'fenugreek',
      'cook',
      'till',
      'fenugreek',
      'soft',
      'cooking'],
     ['add', 'tomatoes'],
     ['turmeric', 'powder'],
     ['sambar', 'powder'],
     ['coriander', 'powder'],
     ['salt'],
     ['mix', 'cook', 'minute', 'minute', 'add', 'tamarind', 'water'],
     ['remaining', 'water', 'let', 'boil', 'boiling', 'minutes'],
     ['add',
      'dal',
      'mix',
      'cook',
      'sambar',
      'garnish',
      'coriander',
      'serve',
      'sundkai',
      'methi',
      'sambar',
      'rice'],
     ['beetroot', 'thoran', 'papad', 'dinner'],
     ['begin', 'making', 'cabbage'],
     ['spinach'],
     ['pomegranate', 'slaw', 'lemon', 'dressing'],
     ['whisk', 'dressing', 'bowl', 'combine', 'lemon', 'juice'],
     ['olive', 'oil'],
     ['salt',
      'pepper',
      'aside',
      'using',
      'fork',
      'emulsify',
      'lemon',
      'oil',
      'add',
      'vegetables',
      'fruits',
      'mix'],
     ['check', 'seasonings'],
     ['require', 'add', 'mix', 'again', 'serve', 'cabbage'],
     ['spinach'],
     ['carrot', 'pineapple', 'orange', 'juice'],
     ['begin', 'making', 'matar', 'samosa'],
     ['mix',
      'half',
      'salt',
      'flour',
      'rub',
      'ghee',
      'vegetable',
      'shortening',
      'flour',
      'mixture'],
     ['add', 'water'],
     ['knead', 'soft', 'firm', 'dough', 'roll', 'dough', 'ball'],
     ['pan'],
     ['warm', 'ghee'],
     ['up',
      'season',
      'salt',
      'cook',
      'minutes',
      'set',
      'peas',
      'mixture',
      'aside',
      'cool',
      'easier',
      'handle',
      'shaping',
      'samosas',
      'make',
      'flour',
      'paste'],
     ['combining', 'flour', 'water', 'to', 'begin', 'making', 'samosa', 'cases'],
     ['out', 'circle', 'halves', 'running', 'knife', 'center'],
     ['ges', 'open', 'cone'],
     ['dollop',
      'dhaniya',
      'pudina',
      'chutney',
      'khajoor',
      'imli',
      'chutney',
      'tomato',
      'ketchup'],
     ['masala', 'chai', 'evening'],
     ['begin',
      'eggless',
      'chocolate',
      'cakes',
      'raspberry',
      'cream',
      'cheese',
      'frosting'],
     ['sieve', 'flour'],
     ['cocoa', 'powder'],
     ['baking', 'powder'],
     ['baking', 'soda', 'aside', 'take', 'bowl', 'whisk', 'butter', 'minutes'],
     ['sure',
      'light',
      'fluffy',
      'add',
      'vegetable',
      'oil',
      'sugar',
      'whisk',
      'minutes',
      'further'],
     ['add',
      'vanilla',
      'extract',
      'milk',
      'half',
      'whisk',
      'in',
      'double',
      'boiling',
      'method',
      'melt',
      'dark',
      'chocolate'],
     ['peaks',
      'whisk',
      'low',
      'pace',
      'peaks',
      'as',
      'whisk',
      'add',
      'sugar',
      'vanilla',
      'extract',
      'mix',
      'well',
      'grind',
      'raspberries',
      'add',
      'cream',
      'cheese',
      'mixture',
      'fill',
      'icing',
      'piping',
      'bag',
      'favourite',
      'nozzle',
      'icing',
      'cake',
      'after',
      'icing',
      'cream',
      'cheese',
      'place',
      'raspberry',
      'serve',
      'it',
      'serve',
      'eggless',
      'chocolate',
      'cakes',
      'raspberry',
      'cream',
      'cheese',
      'frosting',
      'evening',
      'snack',
      'dessert',
      'party'],
     ['begin', 'making', 'kerala', 'palada', 'pradhaman'],
     ['milk', 'add', 'it', 'sprinkle', 'cardamom', 'powder'],
     ['mix',
      'well',
      'serve',
      'hot',
      'cold',
      'ways',
      'es',
      'delicious',
      'serve',
      'kerala',
      'palada',
      'pradhaman',
      'meal',
      'keerai',
      'sambar'],
     ['rice'],
     ['begin', 'making', 'tindora', 'sambharo', 'instant', 'tendli', 'pickle'],
     ['serve', 'tindora', 'sambharo', 'dals'],
     ['oil', 'sesame', 'oil', 'mustard', 'oil', 'better', 'e'],
     ['begin', 'making', 'chettinad', 'style', 'chicken', 'roast'],
     ['pan'],
     ['ium', 'flame', 'add', 'onions'],
     ['coconut'],
     ['ginger',
      'garlic',
      'cook',
      'till',
      'onions',
      'soft',
      'coconut',
      'turn',
      'light',
      'brown',
      'colour',
      'turn',
      'flame',
      'grind',
      'coarse',
      'paste',
      'little',
      'water',
      'mixer',
      'grinder',
      'the',
      'step',
      'marinate',
      'chicken',
      'mixing',
      'bowl'],
     ['add', 'onion', 'coconut', 'paste'],
     ['yogurt'],
     ['kalonji'],
     ['black', 'peppercorn'],
     ['chillies'],
     ['cinnamon', 'stick'],
     ['cloves'],
     ['s', 'dry', 'roast', 'minutes'],
     ['let', 'cool', 'bit'],
     ['add',
      'thes',
      'spices',
      'mixer',
      'grinder',
      'grind',
      'powder',
      'heat',
      'ghee',
      'pan',
      'add',
      'chettinad',
      'spice',
      'mix',
      'ghee',
      'seconds'],
     ['add', 'chicken', 'marinade', 'mix'],
     ['add', 'tamarind', 'pulp'],
     ['masala', 'aside', 'heat', 'grill', 'pan', 'high', 'heat'],
     ['smoking',
      'hot',
      'place',
      'chicken',
      'leg',
      'it',
      'let',
      'nice',
      'crispy',
      'skin',
      'cooking',
      'minutes',
      'side'],
     ['turn',
      'heat',
      'chicken',
      'piece',
      'plate',
      'the',
      'final',
      'step',
      'temper',
      'chicken',
      'tempering'],
     ['garlic',
      'fry',
      'till',
      'golden',
      'brown',
      'turn',
      'flame',
      'to',
      'assemble',
      'chettinad',
      'style',
      'chicken',
      'roast'],
     ['serving', 'plate'],
     ['place', 'chicken', 'leg'],
     ['serve',
      'chettinad',
      'style',
      'chicken',
      'roast',
      'party',
      'starter',
      'kerala',
      'style',
      'appam',
      'without',
      'yeast',
      'mangalorean',
      'neer',
      'dosa',
      'savory',
      'rice',
      'coconut',
      'crepe',
      'complete',
      'meal'],
     ['slit',
      'outer',
      'layer',
      'remove',
      'outer',
      'white',
      'layer',
      'pressure',
      'cooker'],
     ['add', 'salt'],
     ['ium', 'flame'],
     ['add', 'oil', 'oil', 'hot'],
     ['add', 'onions', 'saute', 'translucent'],
     ['tomatoes'],
     ['onion',
      'tomatoes',
      'allow',
      'cool',
      'transfer',
      'onion',
      'tomato',
      'mixture',
      'mixer',
      'jar'],
     ['smooth', 'paste', 'transfer', 'bowl', 'set', 'aside'],
     ['mixer', 'grinder'],
     ['coconut'],
     ['stock'],
     ['grind', 'smooth', 'paste', 'aside', 'heat', 'pan', 'oil'],
     ['ium', 'heat'],
     ['curry', 'leaves', 'splutter', 'sizzle'],
     ['garlic',
      'cloves',
      'saute',
      'till',
      'garlic',
      'turn',
      'golden',
      'once',
      'garlic',
      'turns',
      'golden'],
     ['salt'],
     ['turmeric', 'powder'],
     ['chilli', 'powder'],
     ['ium', 'heat'],
     ['s',
      'now',
      'add',
      'coconut',
      'mixture',
      'stir',
      'well',
      'cook',
      'low',
      'heat',
      'thickens',
      'adjust',
      'seasoning',
      'consistency',
      'palakottai',
      'kuzhambu',
      'liking',
      'turn',
      'flame',
      'garnish',
      'coriander',
      'leaves',
      'serve',
      'palakottai',
      'kuzhambu',
      'raw',
      'jackfruit',
      'poriyal'],
     ['rice', 'thayir', 'semiya', 'hearty', 'weekday', 'lunch'],
     ['start', 'brinjal', 'bharta'],
     ['coming', 'out', 'bake', 'eggplants', 'aside', 'till', 'soft', 'cooling'],
     ['let', 'cook', 'seconds', 'seconds'],
     ['add', 'ginger'],
     ['onion', 'cook', 'till', 'onion', 'soft', 'add', 'tomato'],
     ['turmeric', 'powder'],
     ['chilli'],
     ['coriander', 'powder'],
     ['chili',
      'powder',
      'mix',
      'let',
      'cook',
      'minutes',
      'minutes',
      'add',
      'eggplant'],
     ['butter', 'salt', 'mix'],
     ['slow',
      'heat',
      'let',
      'cook',
      'minutes',
      'turn',
      'gas',
      'garnish',
      'coriander',
      'serve',
      'brinjal',
      'bharta',
      'panchamel',
      'dal'],
     ['spinach', 'raita', 'phulka', 'dinner'],
     ['begin', 'making', 'kesar', 'chai'],
     ['add', 'water', 'saucepan'],
     ['saffron',
      'strands',
      'allow',
      'come',
      'brisk',
      'boil',
      'once',
      'begins',
      'boil',
      'add',
      'tea',
      'leaves',
      'allow',
      'tea',
      'leaves',
      'simmer',
      'water',
      'minute',
      'turn',
      'heat',
      'once',
      'add',
      'tea',
      'leaves'],
     ['allow',
      'brew',
      'little',
      'ensure',
      'tea',
      'stays',
      'light',
      'turn',
      'bitter',
      'stir',
      'milk',
      'chai',
      'allow',
      'rest',
      'seconds',
      'strain',
      'kesar',
      'chai',
      'serve',
      'add',
      'saffron',
      'strands',
      'serving'],
     ['great',
      'look',
      'add',
      'flavor',
      'chai',
      'serve',
      'kesar',
      'chai',
      'samosa',
      'vegetable',
      'bajji',
      'pakora',
      'evening',
      'snack'],
     ['pacha', 'manga', 'pachadi'],
     ['raw', 'mango', 'saucepan', 'tamarind', 'water'],
     ['ium', 'heat', 'mango', 'soft'],
     ['add', 'sambar', 'powder'],
     ['salt', 'mix', 'it', 'let', 'cook', 'thickens', 'thickens'],
     ['curry', 'leaves'],
     ['asafoetida'],
     ['chillies', 'let', 'cook', 'seconds'],
     ['add',
      'tempering',
      'pachdi',
      'mix',
      'serve',
      'pacha',
      'manga',
      'pachadi',
      'keerai',
      'sambar'],
     ['rice'],
     ['chow', 'chow', 'thoran', 'masala', 'tea'],
     ['begin', 'making', 'kodava', 'mudi', 'chekke', 'barthad'],
     ['soak', 'chickpeas', 'overnight', 'pressure', 'cook', 'water'],
     ['water', 'pressure', 'cook', 'whistles', 'release', 'pressure'],
     ['dry'],
     ['cool'],
     ['peel', 'skin', 'chop', 'small', 'pieces', 'grind'],
     ['chillies'],
     ['cloves'],
     ['cinnamon'],
     ['coconut'],
     ['ginger', 'garlic', 'coarse', 'paste'],
     ['aside', 'heat', 'kadai', 'oil'],
     ['turn',
      'heat',
      'transfer',
      'mudi',
      'chekke',
      'barthad',
      'serving',
      'bowl',
      'serve',
      'kodava',
      'mudi',
      'chekke',
      'barthad',
      'palak',
      'tovve',
      'palak',
      'dal',
      'jolada',
      'roti',
      'raita',
      'lunch',
      'meals'],
     ['vegetables'],
     ['ients', 'ready', 'in', 'small', 'mixing', 'bowl'],
     ['ients',
      'crepe',
      'water',
      'crepe',
      'batter',
      'light',
      'fluffy',
      'batter',
      'coat',
      'spoon'],
     ['time',
      'runny',
      'thin',
      'then',
      'strain',
      'crepe',
      'mixture',
      'fine',
      'mesh',
      'sieve',
      'pouring',
      'mix',
      'tall',
      'jug',
      'makes',
      'easier',
      'portion',
      'batter',
      'making',
      'crepes',
      'hot',
      'pan',
      'allow',
      'crepe',
      'batter',
      'rest',
      'room',
      'temperature',
      'minutes',
      'makes',
      'crepes',
      'soft',
      'fluffy',
      'the',
      'step',
      'toss',
      'vegetables',
      'wok',
      'heat',
      'olive',
      'oil',
      'wok',
      'add',
      'broccoli'],
     ['onion'],
     ['bell', 'pepper'],
     ['crunch', 'crepe', 'once'],
     ['turn', 'heat'],
     ['check',
      'salt',
      'adjust',
      'suit',
      'e',
      'stir',
      'parsley',
      'aside',
      'to',
      'begin',
      'making',
      'crepes'],
     ['pan', 'gets', 'hot'],
     ['pour', 'batter', 'cover', 'base'],
     ['cook', 'begins', 'little', 'golden'],
     ['way', 'remaining', 'batter', 'stack', 'crepes', 'other', 'once', 'crepes'],
     ['spread', 'cheese', 'spread', 'inside', 'crepes'],
     ['vegetables',
      'cold',
      'coffee',
      'smoothie',
      'spinach',
      'feta',
      'muffins',
      'breakfast'],
     ['begin', 'making', 'makki', 'methi', 'roti'],
     ['heat', 'water', 'pan', 'till', 'lukewarm', 'in', 'steel', 'mixing', 'bowl'],
     ['add', 'maize', 'flour'],
     ['fenugreek', 'leaves'],
     ['chillies'],
     ['salt',
      'mix',
      'well',
      'add',
      'lukewarm',
      'water',
      'maize',
      'flour',
      'little',
      'little',
      'knead',
      'soft',
      'dough',
      'add',
      'oil',
      'entire',
      'ball',
      'dough',
      'aside',
      'place',
      'tawa',
      'low',
      'heat',
      'let',
      'heat',
      'up',
      'divide',
      'dough',
      'small',
      'balls',
      'roll',
      'ball',
      'sheets',
      'plastic',
      'cling',
      'film',
      'circle',
      'use',
      'hands',
      'help',
      'wet',
      'fingers',
      'small',
      'round',
      'roti',
      'place',
      'roti',
      'tawa',
      'cook',
      'low',
      'flame',
      'sides',
      'brown',
      'spots',
      'appear',
      'prepare',
      'rest',
      'rotis',
      'similar',
      'way',
      'serve',
      'hot',
      'serve',
      'makki',
      'methi',
      'roti',
      'sarson',
      'ka',
      'saag',
      'burani',
      'raita',
      'delicious',
      'weekday',
      'meal'],
     ['begin', 'making', 'cabbage', 'carrot', 'sambharo'],
     ['allow', 'crackle', 'crackles'],
     ['add', 'asafoetida'],
     ['chilies'],
     ['vegetables'],
     ['ients', 'well', 'turn', 'heat', 'low'],
     ['cover', 'pan', 'allow', 'vegetables', 'cook', 'steam', 'minutes'],
     ['want', 'vegetables', 'remain', 'little', 'firm', 'crunchy', 'soft', 'once'],
     ['serve',
      'cabbage',
      'carrot',
      'sambharo',
      'palak',
      'ragi',
      'oats',
      'wheat',
      'thepla',
      'beetroot',
      'sesame',
      'thepla',
      'simple',
      'mean'],
     ['yeast',
      'active',
      'dry',
      'yeast',
      'look',
      'like',
      'small',
      'round',
      'balls',
      'follow',
      'instructions',
      'given',
      'packet',
      'if',
      'fast',
      'action',
      'yeast',
      'looks',
      'like',
      'fine',
      'semolina',
      'bowl',
      'combine',
      'flours'],
     ['add', 'yeast', 'knead', 'dough', 'smooth', 'elastic'],
     ['divide',
      'dough',
      'portions',
      'roll',
      'portion',
      'ball',
      'place',
      'balls',
      'baking',
      'sheet',
      'cover',
      'damp',
      'cloth',
      'let',
      'sit',
      'minutes',
      'while',
      'dough',
      'resting'],
     ['half',
      'hour',
      'bake',
      'pitas',
      'them',
      'time',
      'work',
      'dough',
      'flat',
      'bread',
      'ball'],
     ['dust',
      'little',
      'flour',
      'dust',
      'work',
      'surface',
      'well',
      'rolling',
      'pin'],
     ['baking',
      'sheets',
      'place',
      'inside',
      'oven',
      'bake',
      'pita',
      'dough',
      'puff',
      'rising',
      'forming',
      'bubbles'],
     ['puffing',
      'up',
      'the',
      'process',
      'takes',
      'maximum',
      'minutes',
      'minutes',
      'pita',
      'bread',
      'rise',
      'minutes'],
     ['pita', 'pockets'],
     ['pita', 'sandwich'],
     ['vegetables', 'more'],
     ['begin', 'making', 'andhra', 'style', 'tamati', 'pachadi'],
     ['pan', 'add', 'salt'],
     ['ium',
      'heat',
      'boil',
      'till',
      'loose',
      'water',
      'turn',
      'mushy',
      'paste',
      'alike',
      'stirring',
      'between',
      'tomatoes',
      'boiling'],
     ['pan'],
     ['s', 'crackle'],
     ['chillies', 'fry', 'till', 'fragrant'],
     ['cool',
      'grind',
      'smooth',
      'powder',
      'add',
      'masala',
      'step',
      'tomatoes',
      'mix',
      'well',
      'think',
      'chillies',
      'aren',
      't',
      'spicy'],
     ['chilli',
      'powder',
      'according',
      'preference',
      'step',
      'once',
      'heat',
      'tbsp',
      'oil'],
     ['crackle'],
     ['off',
      'important',
      'makes',
      'sure',
      'chutney',
      'stays',
      'long',
      'time',
      'serve',
      'andhra',
      'style',
      'tamati',
      'pachadi',
      'ghee',
      'masala',
      'dosa',
      'homemade',
      'soft',
      'idlis',
      'breakfast',
      'tea',
      'time',
      'snack'],
     ['herb', 'brown', 'rice'],
     ['add', 'brown', 'rice'],
     ['water'],
     ['chilli', 'flax'],
     ['oregano'],
     ['thyme'],
     ['salt'],
     ['open', 'cooker'],
     ['mix',
      'serve',
      'rice',
      'serve',
      'herb',
      'brown',
      'rice',
      'chicken',
      'curry',
      'sweet',
      'potato',
      'salad',
      'choice'],
     ['begin', 'making', 'aamras'],
     ['roll', 'mangoes', 'palms', 'soften', 'them', 'peel', 'mango'],
     ['transfer', 'pulp', 'mixer', 'jar'],
     ['mango', 'pulp', 'jar'],
     ['add',
      'water',
      'adjust',
      'sugar',
      'liking',
      'depending',
      'sweetness',
      'mango',
      'pulse',
      'smooth',
      'fridge',
      'time',
      'serve',
      'cold',
      'meal',
      'serve',
      'delicious',
      'aamras',
      'puri'],
     ['batata',
      'nu',
      'shaak',
      'mumbai',
      'style',
      'masala',
      'khichia',
      'churi',
      'enjoy',
      'breakfast',
      'evening',
      'snack'],
     ['love', 'it'],
     ['begin'],
     ['rajasthani', 'khooba', 'roti'],
     ['mixing', 'bowl'],
     ['add', 'flour'],
     ['salt'],
     ['water'],
     ['ball',
      'dough',
      'roll',
      'ball',
      'dust',
      'counter',
      'roll',
      'dough',
      'ball',
      'round',
      'roti',
      'spread',
      'little',
      'ghee',
      'roti',
      'place',
      'tawa',
      'seconds'],
     ['hold', 'roti', 'tong'],
     ['sides', 'once'],
     ['makai', 'wali', 'bhindi', 'pyaz', 'ki', 'sabzi'],
     ['begin', 'making', 'peerkangai', 'thogayal'],
     ['pan', 'add', 'cut', 'peels', 'ridge', 'gourd'],
     ['soft'],
     ['ients'],
     ['ridge', 'gourd', 'peels'],
     ['rice', 'ghee'],
     ['oatmeal', 'pakora'],
     ['cook', 'dalia', 'wash', 'panel', 'aside', 'pressure', 'cooker'],
     ['add', 'water'],
     ['little', 'salt'],
     ['oil'],
     ['pak', 'bowl', 'aside', 'cool', 'cooling'],
     ['add', 'gram', 'flour'],
     ['onion'],
     ['coriander'],
     ['chillies'],
     ['salt'],
     ['turmeric', 'powder'],
     ['cumin'],
     ['celery'],
     ['oil'],
     ['add',
      'little',
      'oatmeal',
      'batter',
      'cook',
      'golden',
      'brown',
      'crisp',
      'sides',
      'serve',
      'fry',
      'pakodas',
      'pan',
      'serve',
      'dalia',
      'pakora',
      'coriander',
      'mint',
      'chutney',
      'tamarind',
      'chutney',
      'serve',
      'hot',
      'masala',
      'tea',
      'it'],
     ['begin', 'making', 'paneer', 'tikka', 'kathi', 'roll'],
     ['ready',
      'making',
      'paneer',
      'home',
      'use',
      'homemade',
      'paneer',
      'to',
      'paneer',
      'tikka'],
     ['add',
      'cut',
      'paneer',
      'pieces',
      'allow',
      'rest',
      'minutes',
      'heat',
      'oil',
      'ghee',
      'wok',
      'skillet'],
     ['add', 'paneer', 'mixture', 'sauté', 'high', 'heat', 'minutes'],
     ['cook',
      'paneer',
      'marination',
      'thickens',
      'forms',
      'coat',
      'paneer',
      'once'],
     ['hot', 'place', 'katori', 'center', 'paneer', 'tikka', 'pan'],
     ['hot',
      'coal',
      'katori',
      'spoon',
      'little',
      'bit',
      'ghee',
      'coal',
      'begin',
      'emit',
      'smoke',
      'cover',
      'pan',
      'allow',
      'paneer',
      'tikka',
      'smoky',
      'flavours',
      'after',
      'seconds'],
     ['paneer',
      'tikka',
      'ready',
      'the',
      'step',
      'roti',
      'rolls',
      'mixing',
      'bowl',
      'add',
      'flour',
      'salt'],
     ['mix',
      'fingers',
      'add',
      'little',
      'water',
      'time',
      'smooth',
      'dough',
      'once',
      'dough',
      'ready'],
     ['add', 'oil', 'dough', 'knead', 'couple', 'minutes'],
     ['dough', 'flaky', 'texture', 'to', 'wrap', 'kathi', 'rolls'],
     ['ium', 'heat'],
     ['crisp',
      'the',
      'addition',
      'ghee',
      'helps',
      'creating',
      'crisp',
      'texture',
      'rolls',
      'in',
      'small',
      'mixing',
      'bowl'],
     ['combine', 'raw', 'onions'],
     ['capsicum'],
     ['lemon'],
     ['rotis', 'flat', 'surface', 'place', 'portion', 'filling', 'center'],
     ['way',
      'remaining',
      'dough',
      'portions',
      'serve',
      'serve',
      'paneer',
      'tikka',
      'kathi',
      'rolls',
      'snack',
      'tea',
      'weekends'],
     ['appetizer',
      'parties',
      'mumbai',
      'style',
      'tawa',
      'pulao',
      'moong',
      'sprouts',
      'nawabi',
      'kofta',
      'curry',
      'spicy',
      'pack',
      'kid',
      's',
      'lunch',
      'box'],
     ['cheese'],
     ['pot'],
     ['ium', 'heat', 'add', 'butter', 'butter', 'melts'],
     ['rice', 'mixture', 'arrange', 'flour', 'bowl'],
     ['cheese', 'classic', 'marinara', 'sauce', 'ranch', 'evening', 'snack'],
     ['begin',
      'making',
      'spicy',
      'seafood',
      'stew',
      'casserole',
      'tomatoes',
      'lime'],
     ['clean', 'basa', 'fish', 'cut', 'small', 'pieces', 'clean'],
     ['recommend', 'use', 'it', 'now', 'marinate', 'seafood', 'lime', 'juice'],
     ['salt',
      'pepper',
      'powder',
      'minutes',
      'let',
      'marinate',
      'fridge',
      'while',
      'seafood',
      'marinating',
      'fridge'],
     ['prepare', 'stew', 'in', 'wide', 'pan'],
     ['tomatoes', 'cook', 'tomatoes', 'turn', 'mushy', 'add', 'cumin', 'powder'],
     ['add',
      'coconut',
      'milk',
      'continue',
      'cook',
      'minute',
      'two',
      'turn',
      'stove'],
     ['rice', 'weekend', 'lunch', 'dinner'],
     ['begin', 'making', 'spicy', 'chilli', 'garlic', 'noodles'],
     ['cook', 'noodles', 'instructions', 'packet'],
     ['water'],
     ['salt', 'oil', 'put', 'pot', 'water', 'oil', 'salt'],
     ['heat', 'bring', 'rolling', 'boil', 'when', 'boils'],
     ['turn', 'heat'],
     ['packet'],
     ['al',
      'dente',
      'take',
      'care',
      'cook',
      'noodles',
      'turn',
      'mushy',
      'noodles'],
     ['noodles',
      'getting',
      'mushy',
      'stuck',
      'other',
      'aside',
      'later',
      'use',
      'to',
      'begin',
      'making',
      'spicy',
      'seasoning'],
     ['garlic',
      'slices',
      'onions',
      'toss',
      'high',
      'heat',
      'minute',
      'add',
      'capsicum',
      'cook',
      'minute',
      'two',
      'add',
      'soya',
      'sauce'],
     ['chilli', 'sauce'],
     ['vinegar'],
     ['tomato', 'ketchup'],
     ['chilli',
      'mix',
      'well',
      'cook',
      'constant',
      'stirring',
      'seconds',
      'till',
      'sauces',
      'begin',
      'bubble',
      'add',
      'salt'],
     ['next'],
     ['sauce',
      'toss',
      'cot',
      'noodles',
      'serve',
      'spicy',
      'chilli',
      'garlic',
      'noodles',
      'mushroom',
      'chilli',
      'weekend',
      'meal'],
     ['begin', 'making', 'chanar', 'dalna'],
     ['pan', 'heat', 'oil'],
     ['turmeric', 'paste', 'mix'],
     ['rice', 'bengali', 'luchi', 'watermelon', 'smoothie', 'weekend', 'brunch'],
     ['begin', 'making', 'cheesy', 'garlic', 'broccoli', 'nuggets'],
     ['chop',
      'broccoli',
      'florets',
      'melt',
      'nutralite',
      'garlic',
      'oregano',
      'spread',
      'pan'],
     ['add', 'broccoli', 'saute', 'minute', 'softens', 'bit', 'once', 'softens'],
     ['add', 'cheese'],
     ['cheesy', 'garlic', 'mayo'],
     ['crisp',
      'serve',
      'cheesy',
      'garlic',
      'broccoli',
      'nuggets',
      'cocoa',
      'banana',
      'almond',
      'date',
      'smoothie',
      'school',
      'snack',
      'kids',
      'party',
      'appetizer',
      'nutralite',
      'achari',
      'mayo',
      'dip',
      'side'],
     ['begin', 'making', 'bengali', 'mooli', 'aloo', 'ki', 'sabzi'],
     ['saute', 'seconds', 'after', 'seconds'],
     ['add',
      'chilli',
      'onion',
      'cook',
      'till',
      'onion',
      'soft',
      'translucent',
      'onions',
      'translucent'],
     ['ium', 'flame', 'next', 'add', 'spice', 'powders'],
     ['chilli', 'powder'],
     ['turmeric', 'powder'],
     ['rice', 'weekday', 'lunch'],
     ['chakundar', 'sambar'],
     ['tur',
      'dal',
      'cooker',
      'water',
      'cook',
      'till',
      'cities',
      'come',
      'lentils'],
     ['add', 'little', 'water', 'mix', 'well', 'pressure', 'cooker'],
     ['add', 'tamarind', 'water'],
     ['add', 'chakundar', 'mixture'],
     ['tur', 'dal'],
     ['chilies', 'seconds'],
     ['add', 'curry', 'leaves'],
     ['asafoetida',
      'turn',
      'gas',
      'add',
      'tempering',
      'sambar',
      'mix',
      'garnish',
      'coriander',
      'serve',
      'serve',
      'chakundar',
      'sambar',
      'potato',
      'roast',
      'rice',
      'dinner'],
     ['begin', 'making', 'kaddu', 'palak', 'roti'],
     ['remove', 'skin', 'pumpkin', 'grate', 'them', 'bowl'],
     ['add', 'flour'],
     ['pumpkin'],
     ['spinach'],
     ['spice', 'powders'],
     ['balls', 'dust', 'discs', 'flour', 'rolling'],
     ['raita',
      'curry',
      'simple',
      'bowl',
      'curd',
      'prepare',
      'kaddu',
      'palak',
      'roti',
      'serve',
      'dinner',
      'breakfast',
      'corn',
      'onion',
      'raita',
      'dhaniya',
      'pudina',
      'chutney'],
     ['begin', 'making', 'iranian', 'baida', 'curry'],
     ['boil', 'egg', 'water'],
     ['list', 'grind', 'pan'],
     ['cool',
      'grind',
      'smooth',
      'paste',
      'adding',
      'little',
      'water',
      'heat',
      'kadai',
      'oil'],
     ['de', 'shell', 'eggs'],
     ['cut',
      'half',
      'drop',
      'boiling',
      'curry',
      'boil',
      'curry',
      'minutes',
      'switch',
      'off',
      'serve',
      'garnishing',
      'coriander',
      'leaves',
      'serve',
      'iranian',
      'baida',
      'curry',
      'garlic',
      'naan'],
     ['carrots', 'coriander'],
     ['begin', 'making', 'vegan', 'chickpea', 'omelette'],
     ['pre', 'heat', 'prepare', 'batter', 'to', 'batter'],
     ['sieve',
      'chickpea',
      'flour',
      'besan',
      'mixing',
      'bowl',
      'remove',
      'lumps',
      'to'],
     ['add', 'ginger', 'garlic', 'paste'],
     ['ajwain'],
     ['asafoetida'],
     ['turmeric', 'powder'],
     ['baking', 'soda'],
     ['onion'],
     ['tomato'],
     ['chillies'],
     ['coriander', 'leaves', 'salt', 'mix', 'combine', 'next'],
     ['stir', 'coconut', 'milk', 'or', 'soy', 'almond', 'milk'],
     ['add', 'water', 'batter', 'smooth', 'spreadable'],
     ['watery', 'runny', 'when', 'griddle', 'hot'],
     ['ges', 'omelette', 'cook', 'minutes'],
     ['when', 'looks', 'golden'],
     ['flip',
      'omelette',
      'cook',
      'minutes',
      'serve',
      'hot',
      'peanut',
      'carrot',
      'chutney'],
     ['begin',
      'making',
      'ulundu',
      'kozhukattai',
      'uppu',
      'kozhukattai',
      'urad',
      'dal',
      'modak'],
     ['combine', 'dal'],
     ['chillies'],
     ['ginger'],
     ['cool'],
     ['add',
      'little',
      'rice',
      'flour',
      'time',
      'stirring',
      'till',
      'ulundu',
      'kozhukattai',
      'mixture',
      'comes',
      'together',
      'turn',
      'heat',
      'transfer',
      'rice',
      'flour',
      'mixture',
      'bowl',
      'cover',
      'muslin',
      'cloth',
      'aside',
      'minutes',
      'help',
      'ulundu',
      'kozhukattai',
      'dough',
      'come',
      'together',
      'with',
      'little',
      'oil'],
     ['wet',
      'muslin',
      'cloth',
      'prevent',
      'dough',
      'drying',
      'out',
      'take',
      'lemon',
      'size',
      'portion',
      'ulundu',
      'kozhukattai',
      'dough'],
     ['ball', 'press', 'dough', 'palm', 'hands', 'flat', 'dough'],
     ['dumpling', 'tea', 'time', 'snack'],
     ['rice'],
     ['ready',
      'marinate',
      'chicken',
      'cubes',
      'combining',
      'chicken',
      'spring',
      'onions'],
     ['garlic'],
     ['chilies'],
     ['soya', 'sauce'],
     ['lemon', 'juice'],
     ['black', 'pepper'],
     ['chili', 'powder'],
     ['cumin'],
     ['chilli', 'sauce'],
     ['sweet', 'spicy', 'sauce'],
     ['salt',
      'mixing',
      'bowl',
      'allow',
      'chicken',
      'marinate',
      'minutes',
      'you',
      'boil',
      'noodles',
      'instruction',
      'packet',
      'drain',
      'aside',
      'toss',
      'noodles',
      'oil',
      'prevent',
      'sticking',
      'together',
      'to',
      'mayonnaise',
      'sauceprepare',
      'mayo',
      'sauce',
      'mixing',
      'mayonnaise'],
     ['tomato', 'ketchup'],
     ['rice'],
     ['heat', 'oil', 'wok', 'oil', 'hot', 'add', 'ginger', 'garlic'],
     ['slit', 'chillies', 'saute', 'till', 'raw', 'smell', 'disappears'],
     ['low', 'flame', 'ginger', 'garlic', 'char', 'now', 'add', 'carrots'],
     ['saute', 'minute'],
     ['add', 'vegetables'],
     ['ium', 'heat', 'once', 'bit', 'tender'],
     ['add', 'salt'],
     ['spice', 'powder'],
     ['veggies', 'mix', 'turn', 'heat', 'aside', 'to', 'chicken', 'in', 'wok'],
     ['well', 'sprinkle', 'bit', 'water', 'want'],
     ['rice', 'pot', 'meal', 'lunch'],
     ['dessert',
      'like',
      'coconut',
      'tapioca',
      'pudding',
      'spicy',
      'strawberry',
      'sauce',
      'cinnamon',
      'rice',
      'pudding',
      'cherry',
      'compote'],
     ['tofu', 'casserole'],
     ['wash',
      'soak',
      'rice',
      'hour',
      'drain',
      'aside',
      'heat',
      'ghee',
      'saucepan',
      'gets',
      'hot'],
     ['let', 'splutter', 'add', 'onion'],
     ['chilli'],
     ['chili', 'powder'],
     ['turmeric', 'powder'],
     ['coriander', 'powder'],
     ['turn',
      'gas',
      'cooking',
      'open',
      'minutes',
      'garnish',
      'coriander',
      'serve',
      'tofu',
      'casserole',
      'spinach',
      'raita'],
     ['kachumbar', 'salad', 'papad', 'day', 'dinner'],
     ['begin', 'making', 'masala', 'sandwich', 'rocket', 'leaves'],
     ['masala', 'sandwich', 'heat', 'oil', 'wok', 'kadai'],
     ['add', 'onions'],
     ['ginger'],
     ['garlic'],
     ['ium', 'heat', 'soft', 'tender', 'add', 'tomatoes'],
     ['potatoes'],
     ['paneer',
      'pav',
      'bhaji',
      'masala',
      'stir',
      'combine',
      'check',
      'salt',
      'spices',
      'adjust',
      'suit',
      'e',
      'cover',
      'pan'],
     ['turn', 'heat', 'low', 'cook', 'minutes'],
     ['spread', 'bombay', 'masala', 'mixture', 'half'],
     ['sandwich',
      'holds',
      'together',
      'serve',
      'masala',
      'sandwich',
      'rocket',
      'leaves',
      'cutting',
      'half',
      'serve',
      'hot',
      'strawberry',
      'yogurt',
      'lassi',
      'coffee',
      'banana',
      'oats',
      'smoothie',
      'breakfast'],
     ['begin', 'making', 'tomato', 'relish'],
     ['pan', 'add', 'onions'],
     ['garlic', 'saute', 'translucent', 'add', 'tomatoes'],
     ['mushy', 'cook', 'liquid', 'evaporating'],
     ['add', 'mustard'],
     ['black', 'pepper', 'powder'],
     ['oregano'],
     ['sugar', 'salt', 'cook', 'sauce', 'comes', 'together', 'turn', 'heat'],
     ['let',
      'tomato',
      'relish',
      'cool',
      'store',
      'fridge',
      'glass',
      'container',
      'serve',
      'tomato',
      'relish',
      'dip',
      'use',
      'sauce',
      'making',
      'sandwiches',
      'rolls'],
     ['begin',
      'making',
      'khandeshi',
      'dubuk',
      'vade',
      'gram',
      'flour',
      'dumpling',
      'curry'],
     ['ium', 'heat'],
     ['coriander',
      'leaves',
      'saute',
      'minutes',
      'turn',
      'heat',
      'allow',
      'mixture',
      'cool',
      'blitz',
      'mixer',
      'water',
      'smooth',
      'creamy',
      'mixture',
      'dumpling',
      'kandeshi',
      'vademix',
      'besan',
      'flour'],
     ['chilli', 'powder'],
     ['ium', 'heat'],
     ['drizzle',
      'oil',
      'drop',
      'spoonful',
      'batter',
      'cavity',
      'allow',
      'bhadi',
      's',
      'cook',
      'minutes'],
     ['ium', 'heat'],
     ['drizzle', 'oil'],
     ['chilli', 'powder'],
     ['turmeric', 'powder'],
     ['coriander', 'powder'],
     ['transfer',
      'khandeshi',
      'dubuk',
      'vade',
      'serving',
      'bowl',
      'serve',
      'hot',
      'serve',
      'khandeshi',
      'dubuk',
      'vade',
      'gram',
      'flour',
      'dumpling',
      'curry',
      'jowar',
      'atta',
      'roti',
      'aloo',
      'palak',
      'sabzi',
      'everyday',
      'simple',
      'indian',
      'lunch'],
     ['begin', 'making', 'radish', 'soup'],
     ['wash'],
     ['peel', 'slice', 'radish', 'in', 'saucepan'],
     ['add', 'radish', 'slices'],
     ['garam', 'masala', 'powder'],
     ['water', 'pepper', 'boil', 'soup', 'minutes'],
     ['minutes', 'serve', 'soup', 'hot'],
     ['enjoy', 'mid', 'snack', 'sip', 'lunch', 'dinner', 'winters'],
     ['begin', 'making', 'mango', 'donut', 'cake', 'chocolate', 'glaze'],
     ['preheat',
      'oven',
      'deg',
      'c',
      'grease',
      'donut',
      'doughnut',
      'pan',
      'oil',
      'line',
      'butter',
      'paper',
      'sift',
      'maida',
      'baking',
      'powder'],
     ['baking', 'soda', 'salt', 'bowl', 'mix', 'aside', 'in', 'mixing', 'bowl'],
     ['add', 'oil'],
     ['dry', 'flour', 'mixture'],
     ['oven',
      'degree',
      'c',
      'mango',
      'donut',
      'cake',
      'oven',
      'let',
      'cool',
      'bit',
      'while',
      'mango',
      'donut',
      'cake',
      'baking'],
     ['add',
      'chocolates',
      'cream',
      'saucepan',
      'melt',
      'dark',
      'chocolate',
      'smooth',
      'chocolate',
      'glaze',
      'once',
      'mango',
      'donut',
      'cake'],
     ['hold', 'mango', 'donut', 'cake', 'upside'],
     ['mushrooms', 'onion', 'parmesan', 'herbs'],
     ['tea', 'sandwiches', 'adrak', 'chai', 'tea', 'party'],
     ['begin', 'making', 'matar', 'paneer', 'kachori', 'korma'],
     ['dough', 'kachori', 'dough', 'kachorimix', 'maida'],
     ['salt'],
     ['long', 'once', 'dough'],
     ['pan'],
     ['add', 'peas'],
     ['salt', 'dry', 'spices', 'including', 'turmeric', 'powder'],
     ['chilli', 'powder'],
     ['coriander', 'powder'],
     ['amchur', 'powder'],
     ['mashing', 'peas', 'between', 'minutes'],
     ['ges', 'pat', 'little', 'flat', 'aside', 'to', 'kachoris'],
     ['oil', 'frying', 'pan', 'oil', 'hot'],
     ['add',
      'kachoris',
      'fry',
      'minutes',
      'turn',
      'golden',
      'brown',
      'crispy',
      'sides',
      'once'],
     ['paper',
      'towel',
      'absorb',
      'excess',
      'oil',
      'place',
      'aside',
      'gravy',
      'korma',
      'dishwe',
      'onion',
      'paste'],
     ['pan'],
     ['add', 'oil'],
     ['let', 'heat'],
     ['add', 'spices', 'including', 'cloves'],
     ['mace'],
     ['cardamom'],
     ['cinnamon', 'stick', 'let', 'crackle', 'minute', 'to', 'pan'],
     ['add', 'turmeric', 'powder'],
     ['chilli', 'powder'],
     ['ginger', 'garlic', 'paste'],
     ['gravy'],
     ['add',
      'cashew',
      'paste',
      'tomato',
      'onion',
      'mixture',
      'stir',
      'uniform',
      'mixture',
      'switch',
      'heat',
      'low'],
     ['add', 'garam', 'masala', 'powder'],
     ['sugar'],
     ['salt', 'let', 'cook', 'minutes', 'low', 'heat'],
     ['coriander',
      'leaves',
      'serve',
      'matar',
      'paneer',
      'kachori',
      'korma',
      'boondi',
      'raita',
      'phulka',
      'weekday',
      'meal',
      'serve',
      'matar',
      'paneer',
      'kachori',
      'korma',
      'dal',
      'makhani'],
     ['palak', 'raita'],
     ['phulka'],
     ['jeera', 'rice', 'kachumber', 'salad', 'meal'],
     ['guests', 'coming', 'home', 'dinner', 'organizing', 'house', 'party'],
     ['begin', 'making', 'zucchini', 'roll', 'lasagne'],
     ['prep', 'zucchini'],
     ['long', 'stripes', 'grill', 'zucchini', 'grill', 'pan'],
     ['grilling', 'zucchini', 'stripes'],
     ['aside',
      'cool',
      'down',
      'stuffing',
      'zucchini',
      'roll',
      'lasagneheat',
      'sauce',
      'pan',
      'oil'],
     ['zucchini', 'bell', 'peppers', 'saute', 'minutes'],
     ['sprinkle', 'seasonings', 'like', 'thyme'],
     ['water', 'saucepan', 'once', 'tomato', 'skin', 'peeling'],
     ['turn', 'heat', 'when', 'skin', 'peel'],
     ['know', 'tomatoes', 'ready', 'peel', 'skin', 'tomatoes'],
     ['chop', 'them', 'do', 'juices', 'tomatoes', 'release'],
     ['tomatoes', 'aside', 'in', 'saucepan'],
     ['garlic',
      'onions',
      'stir',
      'seconds',
      'begins',
      'sizzle',
      'oil',
      'onions',
      'soften',
      'at',
      'stage'],
     ['basil', 'leaves'],
     ['salt'],
     ['sugar'],
     ['ajar',
      'helps',
      'thicken',
      'sauce',
      'evaporating',
      'excess',
      'water',
      'add',
      'salt'],
     ['rest', 'zucchini', 'sheets', 'take', 'baking', 'dish'],
     ['top',
      'zucchini',
      'roll',
      'cheese',
      'bake',
      'zucchini',
      'roll',
      'lasagne',
      'till',
      'cheese',
      'melts'],
     ['remove',
      'oven',
      'serve',
      'hot',
      'serve',
      'zucchini',
      'roll',
      'lasagne',
      'garlic',
      'bread',
      'herb',
      'butter'],
     ['tomatoes', 'delicious', 'meal'],
     ['begin', 'cooking', 'thengai', 'sadam', 'coconut', 'rice'],
     ['oil'],
     ['begins', 'crackle'],
     ['add', 'urad', 'dal'],
     ['channa', 'dal', 'peanuts', 'once', 'dals', 'begins', 'brown'],
     ['add', 'chillies'],
     ['curry', 'leaves', 'ginger', 'stir', 'minutes', 'next'],
     ['add',
      'coconut',
      'kadai',
      'mix',
      'together',
      'care',
      'taken',
      'avoid',
      'moisture',
      'remaining',
      'mixture'],
     ['moisture', 'coconut', 'evaporates', 'stage'],
     ['add', 'coconut', 'oil', 'thengai', 'sadam'],
     ['turn',
      'heat',
      'low',
      'simmer',
      'couple',
      'minutes',
      'turn',
      'heat',
      'serve',
      'thengai',
      'sadam',
      'coconut',
      'rice',
      'hot',
      'potato',
      'roast',
      'papad'],
     ['begin',
      'making',
      'karwar',
      'style',
      'ambade',
      'udid',
      'methi',
      'hog',
      'plum',
      'curry'],
     ['scrape', 'away', 'outer', 'layer', 'bitter', 'gourd'],
     ['bit', 'salt', 'i'],
     ['toss',
      'coat',
      'aside',
      'hour',
      'away',
      'bitterness',
      'bitter',
      'gourd',
      'hour'],
     ['urad', 'dal'],
     ['rice', 'grains'],
     ['peppercorns'],
     ['ient', 'changes', 'colour'],
     ['coconut'],
     ['water',
      'form',
      'smooth',
      'paste',
      'add',
      'paste',
      'water',
      'mix',
      'form',
      'semi',
      'thick',
      'gravy',
      'pour',
      'kadai',
      'now'],
     ['add', 'turmeric', 'powder'],
     ['ambade', 'hog', 'plum', 'amtekai'],
     ['jaggery'],
     ['bitter', 'gourd', 'rings'],
     ['till', 'reaches', 'boiling', 'point', 'starts', 'boiling'],
     ['tadka', 'pan'],
     ['heat', 'bit', 'oil', 'tempering', 'oil', 'hot'],
     ['rice', 'sol', 'kadhi', 'south', 'indian', 'meal'],
     ['cauliflower'],
     ['pre', 'heat', 'oven', 'degree', 'celsius', 'in', 'mixing', 'bowl'],
     ['toss', 'olive', 'oil'],
     ['place',
      'florets',
      'baking',
      'tray',
      'roast',
      'till',
      'soften',
      'about',
      'minutes',
      'tray',
      'oven',
      'add',
      'coriander',
      'powder'],
     ['cauliflower',
      'starter',
      'dish',
      'panchmel',
      'dal',
      'phulkas',
      'weekday',
      'lunch',
      'dinner'],
     ['begin', 'making', 'pain', 'viennois', 'vienna', 'bread'],
     ['bowl'],
     ['combine', 'water'],
     ['sugar'],
     ['frothy', 'add', 'flours'],
     ['beetroot', 'gazpacho', 'soup', 'light', 'dinner'],
     ['begin', 'making', 'crunchy', 'ginger', 'capsicum', 'rice'],
     ['let',
      's',
      'basmati',
      'rice',
      'ready',
      'wash',
      'basmati',
      'rice',
      'running',
      'tap',
      'water',
      'clear',
      'soak',
      'minutes',
      'now'],
     ['big', 'pan'],
     ['rice'],
     ['salt'],
     ['al', 'dente', 'firm', 'bite', 'switch', 'flame'],
     ['basmati', 'rice'],
     ['switch',
      'flame',
      'serve',
      'crunchy',
      'ginger',
      'capsicum',
      'rice',
      'kids',
      'lunch',
      'box',
      'chicken',
      'vegetable',
      'manchurian',
      'interesting',
      'weeknight',
      'dinner'],
     ['cook', 'bengali', 'chirer', 'payesh', 'poha', 'pudding', 'nolen', 'gur'],
     ['almonds', 'cashews', 'salt'],
     ['cook',
      'minute',
      'remove',
      'heat',
      'pour',
      'bengali',
      'chirer',
      'payesh',
      'poha',
      'pudding',
      'nolen',
      'gur',
      'serving',
      'bowl'],
     ['sumptuous',
      'meal',
      'cholar',
      'dal',
      'aar',
      'macher',
      'jhol',
      'jeera',
      'rice',
      'dinner'],
     ['begin', 'making', 'makhan'],
     ['now',
      'lumps',
      'butter',
      'floating',
      'buttermilk',
      'spoon',
      'spatula',
      'collect',
      'white',
      'butter',
      'air',
      'tight',
      'container',
      'store',
      'refrigerator',
      'homemade',
      'white',
      'butter',
      'ready',
      'serve',
      'makhan',
      'aloo',
      'paratha'],
     ['mooli', 'paratha', 'paratha', 'choice'],
     ['rice', 'cakes', 's', 'preparing', 'rice', 'cakes'],
     ['sticky', 'paste', 'cover', 'bowl', 'cling', 'film'],
     ['leaving',
      'open',
      'place',
      'microwave',
      'oven',
      'minutes',
      'high',
      'remove',
      'add',
      'water',
      'mix',
      'repeat',
      'process',
      'once',
      'remove',
      'dough',
      'place',
      'board',
      'muddler'],
     ['oil',
      'board',
      'process',
      'prevent',
      'sticking',
      'pound',
      'dough',
      'minutes',
      'after'],
     ['ing',
      'oil',
      'hand',
      'board',
      'wrap',
      'cling',
      'film',
      'store',
      'fridge',
      'hour',
      'remove',
      'fridge',
      'cut',
      'lengthwise',
      'aside',
      'heat',
      'wok',
      'oil'],
     ['add', 'ginger', 'garlic'],
     ['add', 'water', 'cover', 'let', 'cook', 'meanwhile'],
     ['ients', 'small', 'bowl', 'water', 'aside', 'vegetables'],
     ['pour', 'sauce', 'mixture', 'stirring', 'till', 'thickens', 'at', 'point'],
     ['rice',
      'cakes',
      's',
      'general',
      'tso',
      'cauliflower',
      'satisfying',
      'lunch'],
     ['begin', 'making', 'white', 'pumpkin', 'mor', 'kuzhambu'],
     ['running', 'cooker', 'cold', 'water', 'aside', 'meanwhile'],
     ['whisk', 'curd', 'water', 'bowl', 'smooth', 'sure', 'lumps', 'curd', 'next'],
     ['coconut'],
     ['chilli'],
     ['add', 'turmeric', 'powder'],
     ['white', 'pumpkin'],
     ['ges', 'vessel', 'let', 'mixture', 'boil'],
     ['curd',
      'split',
      'high',
      'heat',
      'turn',
      'flame',
      'moment',
      'froth',
      'step',
      'temper',
      'kuzhambu',
      'in',
      'tempering',
      'pan',
      'tadka',
      'pan'],
     ['add', 'coconut', 'oil', 'oil', 'hot'],
     ['starts', 'crackle'],
     ['add', 'curry', 'leaves'],
     ['rice',
      'menthia',
      'keerai',
      'paruppu',
      'usili',
      'simple',
      'weekday',
      'lunch'],
     ['paneer', 'pea', 'paratha'],
     ['ients', 'first', 'paratha', 'flour', 'add', 'wheat', 'flour'],
     ['salt'],
     ['cook', 'seconds', 'add', 'ginger'],
     ['peas', 'cook', 'half', 'add', 'cottage', 'cheese'],
     ['chilli', 'powder'],
     ['coriander', 'powder'],
     ['mango', 'powder'],
     ['salt',
      'mix',
      'aside',
      'let',
      'cool',
      'down',
      'little',
      'piece',
      'flour',
      'roll',
      'out',
      'place',
      'mixture',
      'paneer',
      'peas',
      'middle',
      'close',
      'sides',
      'coat',
      'dry',
      'flour',
      'vine',
      'again',
      'fat'],
     ['ghee',
      'top',
      'cook',
      'till',
      'turns',
      'golden',
      'brown',
      'sides',
      'turn',
      'gas',
      'serve',
      'paneer',
      'hare',
      'matar',
      'paratha',
      'coriander',
      'mint',
      'chutney',
      'tadka',
      'raita',
      'evening',
      'snack'],
     ['senai', 'pachadi'],
     ['peel',
      'cut',
      'yam',
      'small',
      'pieces',
      'pressure',
      'cooker',
      'add',
      'water'],
     ['add', 'coriander'],
     ['chillies'],
     ['ients', 'given', 'tempering', 'cook', 'seconds', 'bowl'],
     ['add', 'yam', 'mixture'],
     ['vegetable', 'sambar'],
     ['beetroot', 'thoran', 'rice', 'dinner'],
     ['drunken', 'noodles'],
     ['boil',
      'flat',
      'noodles',
      'boil',
      'water',
      'saucepan',
      'add',
      'salt',
      'oil',
      'let',
      'boil',
      'high',
      'heat',
      'water',
      'boils'],
     ['add',
      'noodles',
      'cook',
      'till',
      'soft',
      'care',
      'boil',
      'much',
      'turn',
      'gas',
      'drain',
      'water',
      'noodles',
      'cold',
      'water'],
     ['noodles',
      'stop',
      'cooking',
      'transfer',
      'noodles',
      'bowl',
      'add',
      'oil',
      'mix',
      'hands',
      'aside',
      'drunken',
      'noodle',
      'sauce'],
     ['add', 'brown', 'sugar'],
     ['honey'],
     ['soy', 'sauce'],
     ['fish',
      'sauce',
      'sriracha',
      'sauce',
      'mixing',
      'bowl',
      'add',
      'hot',
      'water'],
     ['mix',
      'aside',
      'heat',
      'oil',
      'pan',
      'add',
      'ginger',
      'garlic',
      'cook',
      'seconds',
      'add',
      'onions',
      'cook',
      'minute',
      'minute',
      'add',
      'capsicum'],
     ['baby', 'corn', 'cook', 'till', 'vegetables', 'soft', 'add', 'salt'],
     ['rice',
      'noodles',
      'mix',
      'it',
      'add',
      'drunken',
      'noodle',
      'sauce',
      'cook',
      'minutes',
      'turn',
      'gas',
      'serve',
      'serve',
      'drunken',
      'noodles',
      'basil',
      'chicken',
      'thai',
      'style',
      'cheese',
      'weekend',
      'dinner',
      'serve',
      'tender',
      'coconut',
      'ice',
      'cream',
      'it'],
     ['prepare', 'chettinad', 'style', 'prawn', 'biryani'],
     ['wash', 'clean', 'prawns'],
     ['collect', 'bowl', 'add', 'ginger', 'garlic', 'paste'],
     ['chili', 'powder'],
     ['turmeric'],
     ['handi',
      'add',
      'spices',
      'wait',
      'till',
      'splutter',
      'now',
      'add',
      'shallots',
      'saute',
      'till',
      'pale',
      'color',
      'add',
      'chili',
      'paste'],
     ['season', 'coriander', 'powder'],
     ['rice',
      'stir',
      'incorporates',
      'other',
      'add',
      'warm',
      'water',
      'double',
      'quantity',
      'rice'],
     ['check',
      'salt',
      'add',
      'dollop',
      'ghee',
      'mix',
      'well',
      'allow',
      'rice',
      'come',
      'boil'],
     ['curry',
      'leaves',
      'serve',
      'chettinad',
      'style',
      'prawn',
      'biryani',
      'hot',
      'raita',
      'wholesome',
      'meal'],
     ['trail', 'mix', 'figs', 'honey'],
     ['mixing', 'bowl'],
     ['combine', 'walnuts'],
     ['figs'],
     ['honey'],
     ['trail',
      'mix',
      'figs',
      'honey',
      'watermelon',
      'smoothie',
      'school',
      'snack',
      'kids'],
     ['begin', 'making', 'murgh', 'malaiwala'],
     ['clean',
      'wash',
      'chicken',
      'pieces',
      'dry',
      'it',
      'soak',
      'kasuri',
      'methi',
      'leaves',
      'th',
      'water',
      'small',
      'bowl',
      'take',
      'big',
      'bowl',
      'add',
      'chicken'],
     ['salt'],
     ['pepper', 'mix', 'well', 'refrigerate', 'hours', 'more', 'after', 'hours'],
     ['add', 'garlic'],
     ['ginger'],
     ['lemon', 'juice'],
     ['pan',
      'ghee',
      'add',
      'cardamom',
      'powder',
      'onions',
      'pan',
      'let',
      'cook',
      'till',
      'onions',
      'soft',
      'translucent',
      'next'],
     ['chicken', 'pieces', 'let', 'cook', 'minutes', 'minutes'],
     ['add', 'cream'],
     ['strain',
      'methi',
      'leaves',
      'sprinkle',
      'gravy',
      'stir',
      'well',
      'let',
      'cook',
      'minute',
      'two',
      'switch',
      'gas'],
     ['cover',
      'pan',
      'let',
      'sit',
      'minutes',
      'serving',
      'serve',
      'murgh',
      'malaiwala',
      'burani',
      'raita'],
     ['begin', 'making', 'paal', 'kesari'],
     ['heat', 'kadai', 'milk', 'low', 'flame', 'starts', 'boil'],
     ['add',
      'saffron',
      'sugar',
      'add',
      'cardamom',
      'like',
      'once',
      'sugar',
      'melts'],
     ['rava', 'parallel', 'avoid', 'lumps', 'cook', 'rava', 'low', 'flame'],
     ['roast', 'cashew', 'nuts'],
     ['serve',
      'paal',
      'kesari',
      'south',
      'indian',
      'meal',
      'tomato',
      'onion',
      'sambar'],
     ['jeera', 'rasam'],
     ['chow', 'chow', 'thoran'],
     ['rice', 'elai', 'vadam'],
     ['begin', 'making', 'spicy', 'lemon', 'chicken', 'kebab'],
     ['wash', 'clean', 'chicken'],
     ['cut', 'cubes', 'to', 'marinate', 'chicken'],
     ['mixing', 'bowl'],
     ['combine',
      'chicken',
      'pieces',
      'bell',
      'peppers',
      'onions',
      'lemon',
      'juice'],
     ['thyme'],
     ['chilli', 'flakes'],
     ['pepper', 'powder'],
     ['olive', 'oil'],
     ['bell', 'pepper'],
     ['onion', 'chunk'],
     ['heat', 'grill', 'pan'],
     ['serve', 'spicy', 'lemon', 'chicken', 'kebab'],
     ['tzatziki', 'dip'],
     ['onions', 'chutney', 'interesting'],
     ['paneer', 'butter', 'masala', 'biryani'],
     ['wash', 'rice', 'soak', 'minutes', 'rice', 'biryani'],
     ['heat', 'ghee', 'pressure', 'cooker', 'add', 'cardamom'],
     ['long'],
     ['star', 'anise'],
     ['cinnamon'],
     ['black', 'pepper'],
     ['bay', 'leaves'],
     ['cook', 'minute', 'add', 'rice'],
     ['salt'],
     ['onions'],
     ['coriander', 'leaves'],
     ['mint', 'mix', 'well', 'aside', 'paneer', 'butter', 'masala', 'biryani'],
     ['heat', 'oil', 'pan', 'add', 'cumin'],
     ['cardamom'],
     ['chillies'],
     ['ginger'],
     ['garlic'],
     ['onion', 'cook', 'till', 'onion', 'turns', 'brown', 'add', 'tomatoes'],
     ['salt',
      'turn',
      'gas',
      'allow',
      'cool',
      'mixture',
      'mixer',
      'grinder',
      'grind',
      'paste',
      'heat',
      'oil',
      'pan',
      'add',
      'spices',
      'cook',
      'minute'],
     ['add', 'cumin', 'powder'],
     ['garam', 'masala', 'powder'],
     ['chili', 'powder'],
     ['honey'],
     ['kasoori', 'methi', 'let', 'cook', 'low', 'flame', 'gravy', 'boils'],
     ['add', 'cottage', 'cheese'],
     ['cream', 'mix', 'it', 'turn', 'gas', 'minute', 'biryani'],
     ['onions'],
     ['mint', 'leaves'],
     ...]



# recipe embeddings using gensim


```python
#training fastText model on recipe
from gensim.models import FastText
from gensim.test.utils import datapath
from gensim.utils import tokenize
from gensim import utils

model_indianfood_fasttext = FastText(size=100, window=5, min_count=5, workers=4,sg=1)
model_indianfood_fasttext.build_vocab(sentences=all_Sentences)
total_examples = model_indianfood_fasttext.corpus_count
print("training model!")
model_indianfood_fasttext.train(sentences=all_Sentences, total_examples=total_examples, epochs=5)
print("saving model!")
model_indianfood_fasttext.save("models/model_indianfood_fasttext.model")
```

    training model!
    saving model!



```python
#model_indianfood_fasttext.save("models/model_indianfood_fasttext.model")
total_words1 = model_indianfood_fasttext.corpus_total_words
print(total_words1,"words in Recipe based model Corpus!")

model_indianfood_fasttext.wv.most_similar("daal sabji")
```

    327995 words in Recipe based model Corpus!





    [('sukha', 0.9495968222618103),
     ('sukhe', 0.9443870186805725),
     ('sabji', 0.9418803453445435),
     ('vagharela', 0.9411600232124329),
     ('phalguni', 0.938843309879303),
     ('khichia', 0.9387450814247131),
     ('khooba', 0.9372038245201111),
     ('rabodi', 0.9343590140342712),
     ('kanghou', 0.9335325956344604),
     ('sukhi', 0.9331086277961731)]




```python
model_indianfood_fasttext.wv.most_similar("paneer")
```




    [('seer', 0.8310918807983398),
     ('sheer', 0.8263018727302551),
     ('neer', 0.8107001185417175),
     ('bhurji', 0.7945016622543335),
     ('pani', 0.7879313826560974),
     ('pao', 0.7869946956634521),
     ('kulcha', 0.7759062647819519),
     ('macher', 0.7755663394927979),
     ('beer', 0.7722712755203247),
     ('tandoori', 0.7717341780662537)]




```python
model_indianfood_fasttext.wv.most_similar("tandoori")
```




    [('tikka', 0.9459533095359802),
     ('hari', 0.9128848314285278),
     ('kulcha', 0.906612753868103),
     ('bhurji', 0.9024192690849304),
     ('shikampuri', 0.8992996215820312),
     ('katori', 0.8986305594444275),
     ('kulchas', 0.893599271774292),
     ('achari', 0.8933065533638),
     ('shawarma', 0.8896741271018982),
     ('haleem', 0.8863043189048767)]




```python
#defined functions to get to embeddings for recipes
def getSentenceEmbedding(sentence):
    countFound = 0
    embeddingList = []
    for wordx in sentence:
        try:
            vector1 = model_indianfood_fasttext.wv[wordx]
            #print("word",wordx, vector1[:3])
            embeddingList.append(vector1)
            countFound+=1
        except:
            continue;
    sumEmbeddings = sum(embeddingList)
    return np.true_divide(sumEmbeddings, countFound)  

def getRecipeEmbedding(instruction):
    embeddingList = []
    for sentence in instruction:
        embeddingList.append(getSentenceEmbedding(sentence))
    sumEmbeddings = sum(embeddingList)
    return np.true_divide(sumEmbeddings, len(instruction))  
```


```python
df_indianRecipes['recipe_embedding_fasttext'] =  df_indianRecipes.apply(lambda row: getRecipeEmbedding(row['clean_instructions']), axis = 1)
```


```python
#checking out the new column
df_indianRecipes.to_pickle('processed/df_indianRecipes.pkl')
df_indianRecipes.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TranslatedRecipeName</th>
      <th>TranslatedIngredients</th>
      <th>Cuisine</th>
      <th>Course</th>
      <th>Diet</th>
      <th>TranslatedInstructions</th>
      <th>URL</th>
      <th>clean_ingredients</th>
      <th>ingredient_count</th>
      <th>clean_instructions</th>
      <th>recipe_embedding_fasttext</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Masala Karela Recipe</td>
      <td>6 karela (bitter gourd/ pavakkai) - deseeded,s...</td>
      <td>Indian</td>
      <td>Side Dish</td>
      <td>Diabetic Friendly</td>
      <td>to begin making the masala karela recipe,de-se...</td>
      <td>https://www.archanaskitchen.com/masala-karela-...</td>
      <td>[salt, gram flmy besan, turmeric powder haldi,...</td>
      <td>8</td>
      <td>[[begin, making, masala, karela], [karela, sli...</td>
      <td>[-0.016737932, 0.35984662, -0.24503584, -0.063...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Spicy Tomato Rice (Recipe)</td>
      <td>2-1 / 2 cups rice - cooked, 3 tomatoes, 3 teas...</td>
      <td>South Indian Recipes</td>
      <td>Main Course</td>
      <td>Vegetarian</td>
      <td>to make tomato puliogere, first cut the tomato...</td>
      <td>http://www.archanaskitchen.com/spicy-tomato-ri...</td>
      <td>[tomato, bc belle bhat powder, salt, chickpea ...</td>
      <td>10</td>
      <td>[[tomato, puliogere], [cut, tomatoes, mixer, g...</td>
      <td>[0.034362823, 0.29279393, -0.15771821, -0.1977...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ragi Semiya Upma Recipe - Ragi Millet Vermicel...</td>
      <td>1-1/2 cups rice vermicelli noodles (thin),1 on...</td>
      <td>South Indian Recipes</td>
      <td>South Indian Breakfast</td>
      <td>High Protein Vegetarian</td>
      <td>to begin making the ragi vermicelli recipe, fi...</td>
      <td>http://www.archanaskitchen.com/ragi-vermicelli...</td>
      <td>[rice vermicelli noodle thin, pea matar, chill...</td>
      <td>9</td>
      <td>[[begin, making, ragi, vermicelli], [firm, kee...</td>
      <td>[0.025006209, 0.17075203, -0.1810968, -0.09942...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Gongura Chicken Curry Recipe - Andhra Style Go...</td>
      <td>500 grams chicken,2 onion - chopped,1 tomato -...</td>
      <td>Andhra</td>
      <td>Lunch</td>
      <td>Non Vegeterian</td>
      <td>to begin making gongura chicken curry recipe f...</td>
      <td>http://www.archanaskitchen.com/gongura-chicken...</td>
      <td>[gram chicken, chilly slit, turmeric powder ha...</td>
      <td>12</td>
      <td>[[ients, aside, in, small, pan], [ium, heat], ...</td>
      <td>[0.0039616567, 0.19029821, -0.08325197, -0.147...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Andhra Style Alam Pachadi Recipe - Adrak Chutn...</td>
      <td>1 tablespoon chana dal, 1 tablespoon white ura...</td>
      <td>Andhra</td>
      <td>South Indian Breakfast</td>
      <td>Vegetarian</td>
      <td>to make andhra style alam pachadi, first heat ...</td>
      <td>https://www.archanaskitchen.com/andhra-style-a...</td>
      <td>[chana dal, white urad dal, chilly, es ginger ...</td>
      <td>11</td>
      <td>[[andhra, style, alam, pachadi], [chillies], [...</td>
      <td>[0.054038156, 0.27010298, -0.04160401, -0.1178...</td>
    </tr>
  </tbody>
</table>
</div>




```python
#finding similar food-items using trained embeddings only

from numpy import dot
from numpy.linalg import norm

def find_Similar_dish(xx,embeddingToUse):
    #recipe_embedding_fasttext

    a = df_indianRecipes.loc[xx, embeddingToUse]
    orn = df_indianRecipes.loc[xx, "TranslatedRecipeName"]
    #print(orn,"\nGetting most similar dishes based on",embeddingToUse)
    dishtances = {}
    for i in range(len(df_indianRecipes)):
        if i==xx:
            continue;
        try:
            dn = df_indianRecipes.loc[i, "TranslatedRecipeName"]
            b = df_indianRecipes.loc[i, embeddingToUse]
            cos_sim = dot(a, b)/(norm(a)*norm(b))
            if cos_sim not in dishtances.values():
                dishtances[i] = cos_sim
        except:
            continue;
            
    dishtances_2 = {k: v for k, v in sorted(dishtances.items(), key=lambda item: item[1], reverse = True)}
    mostSimilarDishes = []
    countSim = 0
    for el in dishtances_2.keys():
        mostSimilarDishes.append(el)
        countSim+=1
        if countSim==10:
            break;
    return mostSimilarDishes

def list_Similar_dishes(xx, embeddingToUse):
    dn = df_indianRecipes.loc[xx, "TranslatedRecipeName"]
    additionalColumns = ['Cuisine','Course','Diet']
    similarList1 = find_Similar_dish(xx,embeddingToUse)
    simResults1 = []

    allSuggestedDishNames = []
    print("got all similar dishes!")
    for simIndex in similarList1:
        tempRes = []
        dName = df_indianRecipes.loc[simIndex, "TranslatedRecipeName"]
        dishName = " ".join([w for w in dName.split() if w.lower()!='recipe'])
        tempRes.append(dishName)
        dishNameShort = " ".join(dishName.split()[-2:])
        allSuggestedDishNames.append(dishNameShort)
        for col in additionalColumns:
            tempRes.append(df_indianRecipes.loc[simIndex, col])
        simResults1.append(tempRes)
    
    additionalColumns.insert(0,"Dish")
    print(dn)
    return(pd.DataFrame(simResults1, columns = additionalColumns),allSuggestedDishNames)
```


```python
dishNumber = 311
res = list_Similar_dishes(dishNumber, "recipe_embedding_fasttext")
res[0]
```

    got all similar dishes!
    Kashmiri Style Chicken Pulao Recipe





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dish</th>
      <th>Cuisine</th>
      <th>Course</th>
      <th>Diet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Veg Makhanwala - North Indian Mixed Vegetables...</td>
      <td>North Indian Recipes</td>
      <td>Lunch</td>
      <td>Vegetarian</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bengali Doi Maach (Fish In Yogurt Curry)</td>
      <td>Bengali Recipes</td>
      <td>Lunch</td>
      <td>Non Vegeterian</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bhindi Masala Gravy - Ladies Finger In Tomato ...</td>
      <td>North Indian Recipes</td>
      <td>Side Dish</td>
      <td>Vegetarian</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Andhra Style Chicken Fry</td>
      <td>Andhra</td>
      <td>Appetizer</td>
      <td>High Protein Non Vegetarian</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Delicious Methi Chicken Curry - Murgh Methi Curry</td>
      <td>North Indian Recipes</td>
      <td>Dinner</td>
      <td>Non Vegeterian</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Chettinad Muttai Masala - Chettinad Style Egg ...</td>
      <td>Chettinad</td>
      <td>Lunch</td>
      <td>Eggetarian</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Thalapakattu Chicken Biryani</td>
      <td>Tamil Nadu</td>
      <td>Main Course</td>
      <td>Non Vegeterian</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Lugai Gosht - Mutton In Spicy Tomato Onion Gravy</td>
      <td>North Indian Recipes</td>
      <td>Dinner</td>
      <td>Non Vegeterian</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Punjabi Style Egg Curry</td>
      <td>Punjabi</td>
      <td>Lunch</td>
      <td>Eggetarian</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Chettinad Thakkali Sadam (Chettinad Style Toma...</td>
      <td>Chettinad</td>
      <td>Lunch</td>
      <td>Gluten Free</td>
    </tr>
  </tbody>
</table>
</div>




```python
dishNumber = 567
res = list_Similar_dishes(dishNumber, "recipe_embedding_fasttext")
res[0]
```

    got all similar dishes!
    Homemade Easy Gulab Jamun Recipe - Delicious & Tasty





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dish</th>
      <th>Cuisine</th>
      <th>Course</th>
      <th>Diet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Malpua with Rabri (A Spiced Indian Pancake wit...</td>
      <td>Indian</td>
      <td>Dessert</td>
      <td>Vegetarian</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ragi, Wheat &amp; Oat Waffles With Maple Syrup</td>
      <td>Continental</td>
      <td>World Breakfast</td>
      <td>Eggetarian</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Kaju Pista Roll - Cashew Nut Pistachio Fudge</td>
      <td>North Indian Recipes</td>
      <td>Dessert</td>
      <td>Vegetarian</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Basbousa - Middle Eastern Semolina Cake</td>
      <td>Middle Eastern</td>
      <td>Snack</td>
      <td>Vegetarian</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Homemade Whole Wheat Pav / Ladi Pav</td>
      <td>Goan Recipes</td>
      <td>Side Dish</td>
      <td>Vegetarian</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Eggless Chocolate Chip And Honey Cookies</td>
      <td>Continental</td>
      <td>Snack</td>
      <td>Vegetarian</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Eggless Fudgy Chocolate Chunk Brownie</td>
      <td>Continental</td>
      <td>Dessert</td>
      <td>Vegetarian</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Chiroti - Delicious Crispy Layered Sweet Puri</td>
      <td>Indian</td>
      <td>Dessert</td>
      <td>Vegetarian</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Karachi Biscuits</td>
      <td>Indian</td>
      <td>Snack</td>
      <td>Vegetarian</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Strawberry Brownie Bars</td>
      <td>Continental</td>
      <td>Dessert</td>
      <td>Eggetarian</td>
    </tr>
  </tbody>
</table>
</div>


