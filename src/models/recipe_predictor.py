import pickle as pk
import sklearn.metrics
import pandas as pd


def main_function(test_pos_value, test_neg_value):
    """This is the function that predicts 5 ingredients."""
    file_w2v = "src/models/model-w2v.pkl"
    infile = open(file_w2v,'rb')
    model = pk.load(infile)
    #infile.close()
    
    # file_csm = "src/models/model-csm.pkl"
    # df_similarity = load_model(file_csm)

    w_pos = []
    for item in test_pos_value[0].split(", "):
        w_pos.append(str("'" + item + "'"))
    w_neg = []
    for item in test_neg_value[0].split(", "):
        w_neg.append(str("'" + item + "'"))

    result = get_model_w2v(w_pos, w_neg, model)
    my_food_1, my_food_2, my_food_3, my_food_4, my_food_5 = display_chk_boxes(result)

    return my_food_1, my_food_2, my_food_3, my_food_4, my_food_5

def main_function_2(my_food_list):
    """This is the function that predicts 5 or less recipes."""
    choice = len(my_food_list)

    file_tfidf = "src/models/model-tfidf.pkl"
    infile = open(file_tfidf,'rb')
    df_tfidf = pk.load(infile)
    #infile.close()

    if choice == 1:    
        # USER -- Choose 1 result
        my_recipe_list = df_tfidf[(df_tfidf[my_food_list[0]] > 0)].index.tolist()  
        recipe_list = []
        my_range_value = length_recipe_list(my_recipe_list)
        for i in range(my_range_value):
            recipe_list.append('https://www.food.com/recipe/' + str(my_recipe_list[i]))
        recipe_dict = dict(list(enumerate(recipe_list)))
        return recipe_dict

    else:
        # USER -- Choose 2 results   
        my_recipe_list = df_tfidf[(df_tfidf[my_food_list[0]] > 0) & 
                                    (df_tfidf[my_food_list[1]] > 0)
                                    ].index.tolist()  
        
        if len(my_recipe_list) > 0:
            recipe_list = []
            for item in my_recipe_list:
                my_range_value = length_recipe_list(my_recipe_list)
                
                for i in range(my_range_value):
                    recipe_list.append(f'https://www.food.com/recipe/{my_recipe_list[i]}')
            recipe_dict = dict(list(enumerate(recipe_list)))
            return recipe_dict
        else: 
            return {100: "There are no recipies with these ingredients."}

############################################################
# Helper functions begin below #

def get_model_w2v(w1, w2, model):
    """This function returns a wrod2vec model of similar ingredients"""
    try: 
        return model.wv.most_similar(positive=w1, negative=w2)
    except KeyError:
        # To do:
        # Case: the user entered an ingredient that is not in the BOW from corpus
        return 'There are no recipes with this ingredient set.'

def display_chk_boxes(result):
    """This is a function that aids in displaying the ingredients"""
    my_food_1 = result[0][0]
    my_food_2 = result[1][0]
    my_food_3 = result[2][0]
    my_food_4 = result[3][0]
    my_food_5 = result[4][0]
    return my_food_1, my_food_2, my_food_3, my_food_4, my_food_5

def length_recipe_list(my_recipe_list):
  """This is a function that limits the length of the returned recipe list"""
  if len(my_recipe_list) <= 5:
    my_range_value = len(my_recipe_list)
    return my_range_value
  else:
    return 5

 ##################################################################








                

#     except KeyError:
#         # To do:
#         # Case: the user entered an ingredient that is not in the BOW from corpus
#         return 'There are no recipes with this ingredient set.'
        
#         # #try tokenizing the ingredient
#         # not_in_recipies = "cherries"
#         # my_food_list = not_in_recipies.split(' ')
        
#         # for ingredient in my_food_list:
        

#         #   # with the split ingred string, try finding individual ingredients
#         #   if ingredient in df_tfidf.columns:
#         #     print('yes')
#         #     my_recipe_list = df_tfidf[df_tfidf[ingredient] > 0].index.tolist()
#         #     my_range_value = length_recipe_list(my_recipe_list)
#         #     for i in range(my_range_value):
#         #       print(f'https://www.food.com/recipe/{my_recipe_list[i]}')

#     else:
#         # Case: there are no recipes with the 2 chosen ingredients, so find something similar
#         print(f'There are no recipes with this ingredient set. Here are some suggestions:')
#         my_food_2 = result[2][0]
#         my_recipe_list = df_tfidf[df_tfidf[my_food_2] > 0].index.tolist()
#         list_secondary = []
#         for id_ in my_recipe_list:
#             # use the tf-idf cosine similarity to find something similar
#             for column_id in df_similarity.columns:     
#                 # check if the similarity is between values   
#                 if df_similarity.loc[id_,  column_id] > .01:
#                     value = df_similarity.loc[id_,  column_id]
#                     list_secondary.append([column_id, value])
                
#             pri_sec_values_df = pd.DataFrame(list_secondary, columns=['secondary', 'cs_value'])
#             # sort by cosine similarity value 
#             pri_sec_values_df = pri_sec_values_df.sort_values('cs_value')
#             if len(pri_sec_values_df) > 4:
#                 pri_sec_values_df = pri_sec_values_df.tail(5)
#             recipe_list = []
#             for i in range(5):
#                 sec_id = int(pri_sec_values_df.iloc[i]['secondary'])
#                 recipe_list.append(f'https://www.food.com/recipe/{sec_id}')
        
#             return recipe_list



# def load_model(file):
#     """ Load the models from the .pickle file """
#     model_file = open(file, "rb")
#     loaded_model = pickle.load(model_file)
#     model_file.close()
#     return loaded_model