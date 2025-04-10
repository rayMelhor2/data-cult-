from flask import Flask, request, render_template, jsonify, send_from_directory
import os, json, re, shutil
import pytesseract
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import stopwords
from pdf2image import convert_from_path

app = Flask(__name__)
UPLOAD_FOLDER = 'files'
CATEGORIES_FILE = 'cat.json'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.environ["TOKENIZERS_PARALLELISM"] = "false"
nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def load_categories():
    if not os.path.exists(CATEGORIES_FILE):
        with open(CATEGORIES_FILE, 'w', encoding='utf-8') as f:
            json.dump({}, f)
    with open(CATEGORIES_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_text_from_pdf(pdf_path):
    images = convert_from_path(pdf_path, dpi=150)
    result = []
    for img in images:
        text = pytesseract.image_to_string(img.convert('L'), lang='rus')
        result.append(text)
    return ''.join(result)

def save_text_as_json(filename, text):
    json_filename = os.path.splitext(filename)[0] + '.json'
    json_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.relpath(json_filename, UPLOAD_FOLDER))
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump({"text": text}, json_file, ensure_ascii=False, indent=4)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file or not file.filename.lower().endswith('.pdf'):
        return jsonify({'success': False, 'message': 'Загрузите PDF-файл'})

    filename = re.sub(r'[^\w\s.-а-яА-ЯёЁ]', '', file.filename)
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(temp_path)

    text = extract_text_from_pdf(temp_path)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zа-яё0-9\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    processed_text = ' '.join(words)
    text_embedding = model.encode([processed_text])[0]

    categories = load_categories()
    if not categories:
        os.remove(temp_path)
        return jsonify({'success': False, 'message': 'Нет категорий для распределения'})

    category_names = list(categories.keys())
    category_embeddings = model.encode(category_names)
    similarities = util.cos_sim(text_embedding, category_embeddings)[0]
    best_index = similarities.argmax().item()
    assigned_category = category_names[best_index]

    subcategories = categories[assigned_category]["subcategories"]
    if not subcategories:
        os.remove(temp_path)
        return jsonify({'success': False, 'message': f'В категории «{assigned_category}» нет подкатегорий'})

    sub_embeddings = model.encode(subcategories)
    sub_similarities = util.cos_sim(text_embedding, sub_embeddings)[0]
    best_sub_index = sub_similarities.argmax().item()
    assigned_subcategory = subcategories[best_sub_index]

    target_dir = os.path.join(app.config['UPLOAD_FOLDER'], assigned_category, assigned_subcategory)
    os.makedirs(target_dir, exist_ok=True)

    final_path = os.path.join(target_dir, filename)
    os.rename(temp_path, final_path)

    json_filename = os.path.splitext(final_path)[0] + '.json'
    json_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.relpath(json_filename, UPLOAD_FOLDER))
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump({"text": text}, json_file, ensure_ascii=False, indent=4)
    return jsonify({'success': True, 'message': f'Файл сохранён в категорию: {assigned_category}   и емм былаа присвоена под категория: {assigned_subcategory}'})

@app.route('/get_categories', methods=['GET'])
def get_categories():
    return jsonify(load_categories())

@app.route('/add_category', methods=['POST'])
def add_category():
    data = request.json
    category = data.get('category', '').strip()
    if not category:
        return jsonify({'success': False, 'message': 'Пустое имя категории'})

    categories = load_categories()
    if category in categories:
        return jsonify({'success': False, 'message': 'Категория уже есть'})

    categories[category] = {"subcategories": []}
    with open(CATEGORIES_FILE, 'w', encoding='utf-8') as f:
        json.dump(categories, f, ensure_ascii=False, indent=2)
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], category), exist_ok=True)
    return jsonify({'success': True})

@app.route('/add_subcategory', methods=['POST'])
def add_subcategory():
    data = request.json
    category = data.get('category')
    sub = data.get('subcategory')

    categories = load_categories()
    if category not in categories:
        return jsonify({'success': False, 'message': 'Категория не найдена'})
    if sub in categories[category]["subcategories"]:
        return jsonify({'success': False, 'message': 'Подкатегория уже есть'})

    categories[category]["subcategories"].append(sub)
    with open(CATEGORIES_FILE, 'w', encoding='utf-8') as f:
        json.dump(categories, f, ensure_ascii=False, indent=2)
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], category, sub), exist_ok=True)
    return jsonify({'success': True})

@app.route('/delete_category', methods=['POST'])
def delete_category():
    data = request.json
    category = data.get('category')
    categories = load_categories()
    if category not in categories:
        return jsonify({'success': False, 'message': 'Категория не найдена'})

    del categories[category]
    with open(CATEGORIES_FILE, 'w', encoding='utf-8') as f:
        json.dump(categories, f, ensure_ascii=False, indent=2)

    path = os.path.join(app.config['UPLOAD_FOLDER'], category)
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)

    return jsonify({'success': True})

@app.route('/delete_subcategory', methods=['POST'])
def delete_subcategory():
    data = request.json
    category = data.get('category')
    sub = data.get('subcategory')
    categories = load_categories()
    if category in categories and sub in categories[category]["subcategories"]:
        categories[category]["subcategories"].remove(sub)
        with open(CATEGORIES_FILE, 'w', encoding='utf-8') as f:
            json.dump(categories, f, ensure_ascii=False, indent=2)
        sub_path = os.path.join(app.config['UPLOAD_FOLDER'], category, sub)
        if os.path.exists(sub_path):
            shutil.rmtree(sub_path, ignore_errors=True)
        return jsonify({'success': True})
    return jsonify({'success': False, 'message': 'Не удалось удалить'})

@app.route('/delete_file', methods=['POST'])
def delete_file():
    data = request.json
    rel_path = data.get('path')
    abs_path = os.path.join(app.config['UPLOAD_FOLDER'], rel_path)
    if not os.path.exists(abs_path):
        return jsonify({'success': False, 'message': 'Файл не найден'})
    os.remove(abs_path)
    json_path = os.path.splitext(abs_path)[0] + '.json'
    if os.path.exists(json_path):
        os.remove(json_path)

    return jsonify({'success': True})

@app.route('/file_tree', methods=['GET'])
def file_tree():
    def walk(folder):
        items = []
        for name in os.listdir(folder):
            full = os.path.join(folder, name)
            if os.path.isdir(full):
                items.append({
                    'type': 'folder',
                    'name': name,
                    'children': walk(full)
                })
            elif name.lower().endswith('.pdf'):
                items.append({
                    'type': 'file',
                    'name': name,
                    'path': os.path.relpath(full, app.config['UPLOAD_FOLDER'])
                })
        return items
    return jsonify(walk(app.config['UPLOAD_FOLDER']))

@app.route('/move_file', methods=['POST'])
def move_file():
    data = request.json
    src = os.path.join(app.config['UPLOAD_FOLDER'], data['from'])
    dst = os.path.join(app.config['UPLOAD_FOLDER'], data['to'])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.rename(src, dst)

    src_json = os.path.splitext(src)[0] + '.json'
    dst_json = os.path.splitext(dst)[0] + '.json'
    if os.path.exists(src_json):
        os.rename(src_json, dst_json)

    return jsonify({'success': True})

@app.route('/view_pdf/<path:filepath>')
def view_pdf(filepath):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filepath)

@app.route('/search')
def search():
    query = request.args.get('query', '').lower()
    results = []

    def search_folder(folder):
        for name in os.listdir(folder):
            full = os.path.join(folder, name)
            if os.path.isdir(full):
                search_folder(full)
            elif name.lower().endswith('.pdf'):
                rel = os.path.relpath(full, app.config['UPLOAD_FOLDER'])
                json_path = os.path.splitext(full)[0] + '.json'
                match = query in name.lower()
                if os.path.exists(json_path):
                    with open(json_path, encoding='utf-8') as f:
                        data = json.load(f)
                        match |= query in data.get('text', '').lower()
                if match:
                    results.append({
                        'file': name,
                        'path': rel,
                        'json_path': os.path.relpath(json_path, app.config['UPLOAD_FOLDER'])
                    })

    search_folder(app.config['UPLOAD_FOLDER'])
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)