def validate_input(data):
    test_pos_value = []
    test_neg_value = []
    errors = []
    
    EXPECTED_FEATURES = ("w_pos", "w_neg")
    
    if not data:
        errors.append("Form data must not be empty")
    else:
        for feature in EXPECTED_FEATURES:
            if feature not in data:
                errors.append(f"'{feature}' is a required field")
            else:
                try:
                    if feature == "w_pos":
                        test_pos_value.append(data[feature]) 
                    else:
                        test_neg_value.append(data[feature])       
                except ValueError:
                    errors.append(f"Invalid value for field {feature}: '{data[feature]}'")

    return test_pos_value, test_neg_value, errors


    
