from flask import Flask, send_from_directory, render_template, request, abort
from waitress import serve
from src.utils import validate_input
from src.utils import validate_input
from src.models.recipe_predictor import main_function
from src.models.recipe_predictor import main_function_2

app = Flask(__name__, static_url_path="/static")

@app.route("/")
def index():
    """Return the main page."""
    return send_from_directory("static", "index.html")

@app.route("/get_results", methods=["POST"])
def get_results():
    """ Predict ingredients based on the inputs. """
    data = request.form
    print(data)

    test_pos_value, test_neg_value, errors = validate_input(data)

    if not errors:
        my_food_1, my_food_2, my_food_3, my_food_4, my_food_5 = main_function(test_pos_value, test_neg_value)
        return render_template("results_checkboxes.html", predicted=[my_food_1, my_food_2, my_food_3, my_food_4, my_food_5])
    else:
        return abort(400, errors)

@app.route("/get_final_results", methods=["POST"])
def get_final_results():
    """ Predict recipes based on the inputs. """
    selected_ingredients = request.form.getlist("predicted")
    print(selected_ingredients)
    if request.method == "POST":
        predicted_final = main_function_2(selected_ingredients)
        return render_template("results_final.html", predicted=predicted_final)
    else:
        return abort(400, errors)




    # if not errors:
    #     prediction = main_function(test_pos_value, test_neg_value)
    #     return render_template("results.html", predicted=prediction)
    # else:
    #     return abort(400, errors)


if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=5000)
