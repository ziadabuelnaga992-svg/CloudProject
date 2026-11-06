#ندي عربي رجب - 4241238
import os
import re
import json
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "could", "did", "do", "does", "doing", "don't", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself",
    "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "its", "itself", "just", "me", "more",
    "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other",
    "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "were", "weren't", "what", "when", "where", "which", "while", "who", "whom", "why", "with", "won't",
    "would", "you", "your", "yours", "yourself", "yourselves"
}

LEMMATIZATION_DICT = {
    "muscles": "muscle", "workouts": "workout", "exercises": "exercise",
    "curls": "curl", "presses": "press", "squats": "squat",
    "deadlifts": "deadlift", "pullups": "pullup", "pushups": "pushup",
    "raises": "raise", "calories": "calorie", "proteins": "protein",
    "abs": "ab", "legs": "leg", "shoulders": "shoulder",
    "arms": "arm", "chests": "chest", "backs": "back",
    "eating": "eat", "bigger": "big", "stronger": "strong"
}

DEFAULT_DATA = [
    {
        "title": "How to build chest muscles (Bench Press & Push-ups)",
        "category": "Chest & Pecs",
        "question": "how to build chest muscles bench press pushups pec workout upper lower chest",
        "target_muscle": "Chest / Pectorals",
        "answer": "To build chest muscles effectively, focus on compound movements like the Barbell/Dumbbell Bench Press, Incline Dumbbell Press, and Push-ups. Perform 3-4 sets of 8-12 reps with progressive overload. Ensure adequate protein intake (1.6-2.2g per kg of body weight)."
    },
    {
        "title": "How to get bigger biceps and arms",
        "category": "Arms & Biceps",
        "question": "how to get bigger biceps and arms bicep curls workout peak arm size preacher hammer curls",
        "target_muscle": "Biceps / Brachialis",
        "answer": "To grow your biceps, combine Barbell Curls, Incline Dumbbell Curls, and Hammer Curls to target both the long and short heads as well as the brachialis. Train biceps 2 times a week with 3-4 sets of 10-15 reps with controlled tempo."
    },
    {
        "title": "Best exercises for bigger triceps",
        "category": "Arms & Triceps",
        "question": "best exercises for triceps skull crushers rope pushdowns tricep dips arm thickness",
        "target_muscle": "Triceps Brachii",
        "answer": "Triceps make up about 60% of total arm mass. Focus on: 1) Overhead Tricep Extensions, 2) Cable Tricep Rope Pushdowns, and 3) Close-grip Bench Press or Dips. Do 3-4 sets of 10-12 reps."
    },
    {
        "title": "Best strategy to lose belly fat and burn calories",
        "category": "Weight Loss & Fat Burn",
        "question": "how to lose belly fat burn calories weight loss diet cardio shredding caloric deficit",
        "target_muscle": "Full Body / Core",
        "answer": "Targeted fat loss is a myth. To lose belly fat, you must maintain a caloric deficit (eating 300-500 calories below maintenance), combine resistance training to preserve muscle mass, and do moderate cardio (like 10k daily steps or HIIT)."
    },
    {
        "title": "Daily protein intake for muscle growth",
        "category": "Nutrition & Diet",
        "question": "how much protein should i eat daily for bodybuilding muscle growth hypertrophy nutrition",
        "target_muscle": "Nutrition / Recovery",
        "answer": "For optimal muscle hypertrophy, consume between 1.6 to 2.2 grams of protein per kilogram of body weight daily (approx 0.8-1g per pound). High quality sources include chicken breast, eggs, fish, Greek yogurt, whey protein, and beef."
    },
    {
        "title": "Best leg workout for quads and glutes (Squats)",
        "category": "Legs & Lower Body",
        "question": "best leg workout for quads and glutes squats leg press hamstring leg day calf raise",
        "target_muscle": "Quadriceps / Hamstrings / Glutes",
        "answer": "A complete leg day should include: 1) Barbell Squats (Quad/Glute focus), 2) Romanian Deadlifts (Hamstrings/Glutes), 3) Leg Press, and 4) Standing Calf Raises. Strive for deep range of motion and progressive load."
    },
    {
        "title": "How to build a wide back (Deadlifts & Pull-ups)",
        "category": "Back & Posture",
        "question": "how to build a wide back lat pulldown deadlift pull ups lats v taper barbell row",
        "target_muscle": "Latissimus Dorsi / Upper Back",
        "answer": "For back width, prioritize Pull-ups and Lat Pulldowns. For back thickness, do Bent-over Barbell Rows and Deadlifts. Always squeeze your shoulder blades and pull through your elbows."
    },
    {
        "title": "How to get broader 3D shoulders",
        "category": "Shoulders & Delts",
        "question": "best shoulder workout overhead press lateral raises rear delts 3d shoulders military press",
        "target_muscle": "Deltoids (Front, Side, Rear)",
        "answer": "Build 3D shoulders by hitting all 3 heads: 1) Overhead Shoulder Press (front delts), 2) Lateral Raises (side delts for width), and 3) Face Pulls or Reverse Pec Deck (rear delts for 3D look)."
    },
    {
        "title": "How to get visible six pack abs",
        "category": "Abs & Core",
        "question": "how to get six pack abs core workout planks crunches hanging leg raises",
        "target_muscle": "Abdominals / Core",
        "answer": "Abs are revealed through low body fat percentage (typically below 12-14% for men, 18-20% for women). Strengthen the core with Hanging Leg Raises, Cable Woodchoppers, and Weighted Planks 2-3 times a week."
    },
    {
        "title": "Is Creatine safe and how to take it",
        "category": "Supplements",
        "question": "is creatine safe benefits how to take creatine monohydrate dosage strength",
        "target_muscle": "Supplements / Energy",
        "answer": "Creatine Monohydrate increases ATP energy in muscles, boosting strength and power output. Take 3-5 grams daily consistently with plenty of water. No loading phase is required."
    }
]

def fetch_api_data(limit=25):
    api_url = f"https://wger.de/api/v2/exerciseinfo/?language=2&limit={limit}"
    data = []
    try:
        response = requests.get(api_url, timeout=4)
        if response.status_code == 200:
            results = response.json().get("results", [])
            for item in results:
                name = item.get("name", "").strip()
                if not name:
                    continue
                category = item.get("category", {}).get("name", "Exercise")
                muscles = [m.get("name", "") for m in item.get("muscles", []) if m.get("name")]
                all_muscles = ", ".join(muscles) if muscles else "General"
                raw_desc = item.get("description", "")
                clean_desc = re.sub(r'<.*?>', ' ', raw_desc).strip()
                if not clean_desc or len(clean_desc) < 10:
                    clean_desc = f"Perform {name} targeting {all_muscles} with proper form for 3-4 sets."
                
                data.append({
                    "title": name,
                    "category": category,
                    "question": f"{name} {category} {all_muscles} {clean_desc}",
                    "target_muscle": all_muscles,
                    "answer": f"{name} ({category}): {clean_desc}\nTarget: {all_muscles}"
                })
    except Exception:
        pass
    return data

def get_knowledge_base():
    api_items = fetch_api_data()
    knowledge = list(DEFAULT_DATA)
    existing_titles = {item["title"].lower() for item in knowledge}
    for item in api_items:
        if item["title"].lower() not in existing_titles:
            knowledge.append(item)
            existing_titles.add(item["title"].lower())
    return knowledge

class TextPreprocessor:
    def __init__(self):
        self.stop_words = STOP_WORDS
        self.lemma_dict = LEMMATIZATION_DICT

    def clean_text(self, text):
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'<.*?>', ' ', text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def tokenize(self, text):
        if not text:
            return []
        return [w for w in text.split() if w]

    def remove_stopwords(self, tokens):
        return [w for w in tokens if w not in self.stop_words and len(w) > 1]

    def lemmatize_word(self, word):
        if word in self.lemma_dict:
            return self.lemma_dict[word]
        if word.endswith("ies") and len(word) > 4:
            return word[:-3] + "y"
        elif word.endswith("es") and len(word) > 4:
            return word[:-2]
        elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
            return word[:-1]
        elif word.endswith("ing") and len(word) > 5:
            return word[:-3]
        elif word.endswith("ed") and len(word) > 4:
            return word[:-2]
        return word

    def process(self, text):
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        filtered = self.remove_stopwords(tokens)
        lemmatized = [self.lemmatize_word(w) for w in filtered]
        return " ".join(lemmatized)

    def process_with_steps(self, text):
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        filtered = self.remove_stopwords(tokens)
        lemmatized = [self.lemmatize_word(w) for w in filtered]
        return {
            "step_1_raw_text": text,
            "step_2_cleaned_text": cleaned,
            "step_3_tokens": tokens,
            "step_4_without_stopwords": filtered,
            "step_5_lemmatized_roots": lemmatized,
            "step_6_final_processed": " ".join(lemmatized)
        }

#زياد همام عبدالقادر - 42412184
class FitnessChatbot:
    def __init__(self, confidence_threshold=0.15):
        self.preprocessor = TextPreprocessor()
        self.confidence_threshold = confidence_threshold
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.knowledge_base = get_knowledge_base()
        self.train_model()

    def train_model(self):
        corpus = []
        for item in self.knowledge_base:
            combined = f"{item['title']} {item['question']} {item['target_muscle']}"
            corpus.append(self.preprocessor.process(combined))
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def predict(self, user_query, top_k=3):
        steps = self.preprocessor.process_with_steps(user_query)
        processed_query = steps["step_6_final_processed"]
        if not processed_query:
            return {
                "success": False,
                "answer": "Please ask a fitness or workout related question.",
                "confidence": 0.0,
                "top_matches": [],
                "pipeline_steps": steps
            }
            
        query_vector = self.vectorizer.transform([processed_query])
        scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        ranked = np.argsort(scores)[::-1]
        
        top_matches = []
        for idx in ranked[:top_k]:
            sc = float(scores[idx])
            it = self.knowledge_base[idx]
            top_matches.append({
                "title": it["title"],
                "category": it.get("category", "Fitness"),
                "target_muscle": it.get("target_muscle", "Full Body"),
                "similarity_score": round(sc * 100, 1),
                "answer": it["answer"]
            })
            
        best_idx = ranked[0]
        best_score = float(scores[best_idx])
        best_item = self.knowledge_base[best_idx]
        
        feature_names = self.vectorizer.get_feature_names_out()
        query_dense = query_vector.todense().tolist()[0]
        matched_keywords = [(feature_names[i], round(query_dense[i], 3)) for i in range(len(query_dense)) if query_dense[i] > 0]
        matched_keywords.sort(key=lambda x: x[1], reverse=True)
        steps["step_7_query_vector_keywords"] = matched_keywords
        
        if best_score < self.confidence_threshold:
            return {
                "success": False,
                "answer": f"Low confidence match ({round(best_score * 100, 1)}%). Try asking about chest, biceps, back, nutrition, or legs.",
                "confidence": round(best_score * 100, 1),
                "category": best_item.get("category", "General"),
                "target_muscle": best_item.get("target_muscle", "General"),
                "top_matches": top_matches,
                "pipeline_steps": steps
            }
            
        return {
            "success": True,
            "answer": best_item["answer"],
            "title": best_item["title"],
            "category": best_item.get("category", "Fitness"),
            "target_muscle": best_item.get("target_muscle", "General"),
            "confidence": round(best_score * 100, 1),
            "top_matches": top_matches,
            "pipeline_steps": steps
        }

    def respond(self, query):
        res = self.predict(query)
        if not res["success"]:
            return res["answer"]
        return f"[{res['title']}] (Confidence: {res['confidence']}%)\n{res['answer']}"

if __name__ == "__main__":
    bot = FitnessChatbot()
    print("Fitness Chatbot Ready! (Type 'quit' or 'exit' to stop)\n")
    
    test_samples = [
        "How to build bigger chest muscles?",
        "How much protein should I eat daily?",
        "What is the best exercise for biceps?"
    ]
    
    for sample in test_samples:
        print(f"User: {sample}")
        print(f"Bot: {bot.respond(sample)}\n")
        
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        if user_input.strip():
            print(f"Bot: {bot.respond(user_input)}\n")
