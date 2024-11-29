from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import os

app = Flask(__name__)
data_folder = 'tables'
os.makedirs(data_folder, exist_ok=True)

def get_csv_path(table_name):
    return os.path.join(data_folder, f'{table_name}.csv')

def add_record_to_csv(table, model, task_name, value):
    path = get_csv_path(table)
    new_row = pd.DataFrame({'Model': [model], task_name: [value]})
    
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        # Ensure the initially created DataFrame contains the 'Model' column and task column, and is ordered
        df = pd.DataFrame(columns=sorted(['Model', task_name]))

    if model in df['Model'].values:
        df.loc[df['Model'] == model, task_name] = value
    else:
        df = pd.concat([df, new_row], ignore_index=True)
    
    # Reorder the DataFrame columns before saving
    # Ensure the 'Model' column is always the first column
    df = df.sort_index(axis=1)

@app.route('/add', methods=['POST'])
def add_record():
    data = request.json
    add_record_to_csv(data['table'], data['model'], data['task_name'], data['value'])
    return jsonify({'message': 'Record added successfully'}), 201

@app.route('/export/<string:table_name>', methods=['GET'])
def export(table_name):
    path = get_csv_path(table_name)
    if not os.path.exists(path):
        return jsonify({'message': 'Table not found'}), 404
    return send_from_directory(data_folder, f'{table_name}.csv', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=9898, host="0.0.0.0")
    