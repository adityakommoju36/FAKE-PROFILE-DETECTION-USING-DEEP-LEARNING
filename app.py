from flask import Flask, render_template, request
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        required_fields = ['followers_count', 'statuses_count', 'verified', 'account_age_days', 'friends_count', 'default_profile_image']
        for field in required_fields:
            if field not in request.form:
                return render_template('result.html', error=f"Missing field: {field}")

        # Get user inputs safely
        followers = int(request.form.get('followers_count', 0))
        statuses = int(request.form.get('statuses_count', 0))
        verified = int(request.form.get('verified', 0))
        account_age = int(request.form.get('account_age_days', 0))
        friends = int(request.form.get('friends_count', 0))
        default_image = int(request.form.get('default_profile_image', 0))
        
        # Feature calculations
        ER = statuses / (followers + 1e-6)  # Engagement Ratio
        FA = friends / (followers + 1e-6)  # Friend-Follower Ratio
        TFS = statuses / (account_age + 1e-6)  # Tweet Frequency Score
        SPI = verified  # Social Proximity Index (0 or 1)

        # Normalize values
        def normalize(value, min_val, max_val):
            return (value - min_val) / (max_val - min_val + 1e-6)
        
        ER_norm = normalize(ER, 0, 1)
        FA_norm = normalize(FA, 0, 1)
        TFS_norm = normalize(TFS, 0, 1)
        SPI_norm = normalize(SPI, 0, 1)

        # Assign weights
        w1, w2, w3, w4 = 0.25, 0.25, 0.25, 0.25  # Equal weights

        # Compute P_fake Score
        P_fake = w1 * (1 - ER_norm) + w2 * FA_norm + w3 * (1 - SPI_norm) + w4 * TFS_norm
        P_fake = np.clip(P_fake, 0.05, 0.95)

        # Classification
        threshold = 0.6
        pred_class = "Fake" if P_fake > threshold else "Real"

        logger.debug(f"Followers: {followers}, Friends: {friends}, Statuses: {statuses}, Verified: {verified}, Account Age: {account_age}, Default Image: {default_image}")
        logger.debug(f"ER: {ER:.4f}, TFS: {TFS:.4f}, FA: {FA:.4f}, SPI: {SPI}, P_fake: {P_fake:.4f}")

        return render_template('result.html', 
                               prediction=pred_class, 
                               friends=friends,
                               followers=followers,
                               statuses=statuses,
                               verified="Yes" if verified == 1 else "No",
                               account_age=account_age,
                               default_image="Yes" if default_image == 1 else "No",
                               debug_info=f"ER: {ER:.4f}, TFS: {TFS:.4f}, FA: {FA:.4f}, SPI: {SPI}, P_fake: {P_fake:.4f}")
    
    except Exception as e:
        return render_template('result.html', error=f"Error processing request: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
