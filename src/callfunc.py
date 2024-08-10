# sample.py

def callthis():
    print("Function callthis() has been called.")
    # Call the callthat() function from within callthis()
    callthat()

def callthat():
    print("Function callthat() has been called.")

# Calling callthis() will in turn call callthat()
callthis()
