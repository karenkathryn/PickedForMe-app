import pickle
import sklearn.metrics
import pandas as pd


def main_function(test_pos_value, test_neg_value):
    """This is the function that predicts 5 ingredients."""
    file_w2v = "src/models/model-w2v.pkl"
    model = load_model(file_w2v)
    #infile.close()
    
    file_csm = "src/models/model-csm.pkl"
    df_similarity = load_model(file_csm)

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
    """This is the function that predicts 5 or less recipes. The function
    returns a dictionary of 5 recipes or a dictionary with a text statement."""
    # Load the models
    file_tfidf = "src/models/model-tfidf.pkl"
    df_tfidf = load_model(file_tfidf)

    choice = len(my_food_list)

    file_df = "src/models/recipe_data_220k.pkl"
    df = load_model(file_df)
    df = df.set_index('id')
    df['name'] = df.iloc[:, 0].str.replace(r'\ {2,}', " ", regex=True).str.replace(" s ", "'s ").str.upper()

    # USER -- Choose 1 result
    if choice == 1: 
       
        my_recipe_list = df_tfidf[(df_tfidf[my_food_list[0]] > 0)].index.tolist()  
        recipe_list = []
        recipe_name = []
        my_range_value = length_recipe_list(my_recipe_list)
        for i in range(my_range_value):
            recipe_id = my_recipe_list[i]
            recipe_list.append('https://www.food.com/recipe/' + str(recipe_id))
            recipe_name.append( df.loc[recipe_id, 'name'] )

        recipe_dict = dict(list(enumerate(recipe_list)))
        recipe_name = dict(list(enumerate(recipe_name)))
        sim_recipe_dict, sim_name_dict = main_function_3(df, my_recipe_list)
        return '1', recipe_dict, sim_recipe_dict, recipe_name, sim_name_dict

    # USER -- Choose 2 results
    elif choice == 2:   
        my_recipe_list = df_tfidf[(df_tfidf[my_food_list[0]] > 0) & 
                                    (df_tfidf[my_food_list[1]] > 0)
                                    ].index.tolist()  
        if len(my_recipe_list) > 0:
            recipe_list = []
            for item in my_recipe_list:
                my_range_value = length_recipe_list(my_recipe_list)
                
                for i in range(my_range_value):
                    recipe_list.append('https://www.food.com/recipe/' + str(my_recipe_list[i]))
            recipe_dict = dict(list(enumerate(recipe_list)))
            return '2a', recipe_dict, ''
        else: 
            # try individually
            
            recipe_name = []
            recipe_list = []
            for item in my_food_list:
                my_recipe_list = df_tfidf[df_tfidf[item] > 0].index.tolist()  
                my_range_value = length_recipe_list(my_recipe_list)
                for i in range(my_range_value):
                    recipe_id = my_recipe_list[i]
                    recipe_list.append('https://www.food.com/recipe/' + str(recipe_id))
                    recipe_name.append( df.loc[recipe_id, 'name'] )
            recipe_dict = dict(list(enumerate(recipe_list)))
            name_dict = dict(list(enumerate(recipe_name)))
            return '2b', recipe_dict, 'There are no recipes with this choice of ingredients. Here are some suggestions:', name_dict

    else:
        # USER -- Choose 3+ results
        return '3', {100: "That is not a valid option."}, ''

def main_function_3(df, my_recipe_list):
    """The function that finds similar recipes """
    file_csm = "src/models/model-csm.pkl"
    df_similarity = load_model(file_csm)
    
    list_secondary = []
    for id_ in my_recipe_list:
        # use the tf-idf cosine similarity to find something similar
        for column_id in df_similarity.columns:        
            if df_similarity.loc[id_,  column_id] > 0:
                value = df_similarity.loc[id_,  column_id]
                if column_id not in my_recipe_list:
                    list_secondary.append([column_id, value])
                

        pri_sec_values_df = pd.DataFrame(list_secondary, columns=['secondary', 'cs_value'])
        # sort by cosine similarity value 
        pri_sec_values_df = pri_sec_values_df.sort_values('cs_value')
        
        if len(pri_sec_values_df) > 4:
            pri_sec_values_df = pri_sec_values_df.tail(5)
        
        recipe_list = []
        recipe_name = []
        for i in range(5):
            sec_id = int(pri_sec_values_df.iloc[i]['secondary'])
            recipe_list.append('https://www.food.com/recipe/' + str(sec_id))
            recipe_name.append( df.loc[sec_id, 'name'] )

        print(recipe_list)
        recipe_dict = dict(list(enumerate(recipe_list)))
        name_dict = dict(list(enumerate(recipe_name)))
        return recipe_dict, name_dict



############################################################
# Helper functions begin below #
############################################################

def get_model_w2v(w1, w2, model):
    """This function returns a wrod2vec model of similar ingredients"""
    try: 
        return model.wv.most_similar(positive=w1, negative=w2, topn=5)
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


def load_model(file):
    """ Load the models from the .pickle file """
    model_file = open(file, "rb")
    loaded_model = pickle.load(model_file)
    model_file.close()
    return loaded_model

 ##################################################################






