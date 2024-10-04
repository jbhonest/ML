from datetime import datetime

# Get current time
current_time = datetime.now()

# Print only hour and minute
print(current_time.strftime("%H:%M"))
