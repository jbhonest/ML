import joblib

# Load the saved model from the file
model = joblib.load('article_classifier.pkl')

# Sample new article
new_article = """
Brazil’s Senate Economic Affairs Committee approved Gabriel Galipolo’s nomination as central bank governor on Tuesday, setting up a definitive floor vote that’s expected to take place in a matter of hours.
Committee members voted unanimously in favor of Galipolo, who is currently the central bank’s monetary policy director. Galipolo secured approval after telling lawmakers that President Luiz Inacio Lula da Silva indicated he’ll have the freedom to make decisions without political interference.
“Every time the president met with me or called me, it was to say ‘you will be completely at ease with me. I will never ask you anything beforehand, nor will I ever interfere. You will have complete freedom to make decisions in accordance with your judgment. I will always show respect,’” Galipolo said.
Brazilian senators are widely expected to approve Galipolo’s nomination to command monetary policy in Latin America’s largest economy. He would replace Roberto Campos Neto and take the reins of a bank that halted an easing campaign in June and started a cycle of rate hikes in September. He stands to face price challenges including strong growth, higher public spending and a drought— all factors that are keeping inflation forecasts above the 3% target.
The central bank has had autonomy since 2021, meaning Lula can object to policymakers’ decisions but cannot force the board to follow his guidance. At several points during the hearing Galipolo was asked about how he would react if the leftist president were to demand lower rates, with a senator asking him about his “courage.”
Above-target inflation estimates are concerning for the central bank, and there’s a need for more caution in monetary policy, Galipolo told lawmakers. Brazil’s economic growth forecasts have been systematically revised higher due in part to the government’s spending, he said.
The “progressive and distributive” nature of Lula’s fiscal policies is boosting consumption, he said.
A weak currency and strong labor market are other factors driving price forecasts above the central bank’s goal, Galipolo said during his remarks, when he quoted famous economists such as John Maynard Keynes, as well as former UK Prime Minister Winston Churchill and Argentine writer Jorge Luis Borges.
“It’s not the central bank’s job to take risks,” Galipolo said. “The job is to be more conservative and to guarantee that the interest rate is at the adequate level to hit the inflation target.”
Galipolo said the central bank shouldn’t have an “elastic” notion of its 3% goal, as its tolerance range isn’t meant to reduce monetary policy efforts.
"""

# Use the loaded model to predict the category of the new article
predicted_category = model.predict([new_article])

print(predicted_category[0])
