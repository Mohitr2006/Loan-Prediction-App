import os
from flask import Flask, request, render_template, redirect, flash, url_for
import pandas as pd
import joblib
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime, timezone

load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

preprocess = joblib.load("pipeline.pkl")
model = joblib.load("model.pkl")

@app.route('/', methods=["GET","POST"])
def home():

    prediction = None

    if request.method == "POST":
        data = {
            "no_of_dependents": int(request.form["no_of_dependents"]),
            "education": request.form["education"],
            "self_employed": request.form["self_employed"],
            "income_annum": float(request.form["income_annum"]),
            "loan_amount": float(request.form["loan_amount"]),
            "loan_term": int(request.form["loan_term"]),
            "cibil_score": int(request.form["cibil_score"]),
            "residential_assets_value": float(request.form["residential_assets_value"]),
            "commercial_assets_value": float(request.form["commercial_assets_value"]),
            "luxury_assets_value": float(request.form["luxury_assets_value"]),
            "bank_asset_value": float(request.form["bank_asset_value"])
        }

        df = pd.DataFrame([data])

        x = preprocess.transform(df)
        prediction = model.predict(x)[0]

        if prediction == 1:
            prediction = "✅ Approved"
        else:
            prediction = "❌ Rejected"

    return render_template("home.html", prediction=prediction)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/feedback", methods=["POST", "GET"])
def feedback():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        message = request.form.get("message")
        if not name:
            flash("Name cannot be Empty")
            return redirect(url_for("feedback"))
        
        try:
            client["loan_prediction"]["feedback"].insert_one({
                "name": name,
                "email": email,
                "message": message,
                "created_at": datetime.now(timezone.utc)
            })
        except Exception as e:
            print(e)
            flash("Something went wrong. Please try again.")
            return redirect(url_for("feedback"))

        return redirect(url_for("thankyou"))
    
    return render_template("feedback.html")

@app.route("/thankyou")
def thankyou():
    return render_template("thankyou.html")

if __name__ == "__main__":
    app.run(debug=True)