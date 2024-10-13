# app.py

from flask import Flask, request, jsonify
from ufc_model import UFCFightPredictor

app = Flask(__name__)

# Initialize the predictor and load models and cleaned data
predictor = UFCFightPredictor()
predictor.load_cleaned_data()
predictor.load_models()

@app.route('/win_probability', methods=['GET'])
def win_probability():
    fighter_name = request.args.get('fighter')
    if not fighter_name:
        return jsonify({'error': 'Please provide a fighter name.'}), 400
    try:
        prob = predictor.get_win_probability(fighter_name)
        return jsonify({'fighter': fighter_name, 'win_probability': prob})
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 404
    except Exception as e:
        return jsonify({'error': 'An unexpected error occurred.'}), 500

@app.route('/match_probability', methods=['GET'])
def match_probability():
    fighter_A = request.args.get('fighter_A')
    fighter_B = request.args.get('fighter_B')
    if not fighter_A or not fighter_B:
        return jsonify({'error': 'Please provide both fighter_A and fighter_B names.'}), 400
    try:
        prob_A, prob_B = predictor.match_probability(fighter_A, fighter_B)
        return jsonify({
            'fighter_A': fighter_A,
            'probability_A': prob_A,
            'fighter_B': fighter_B,
            'probability_B': prob_B
        })
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 404
    except Exception as e:
        return jsonify({'error': 'An unexpected error occurred.'}), 500

if __name__ == '__main__':
    app.run(debug=True)