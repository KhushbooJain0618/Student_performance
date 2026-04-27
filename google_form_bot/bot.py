from selenium import webdriver
from selenium.webdriver.common.by import By
import random, time

driver = webdriver.Firefox()

FORM_URL = "https://forms.gle/h6UMKpYmQ9Wt8omC9"
driver.get(FORM_URL)

time.sleep(4)

TOTAL_RESPONSES = 80
OUTLIER_PROB = 0.05

for i in range(TOTAL_RESPONSES):

    print("Submitting response", i+1)

    # -------- STUDENT TYPE --------

    student_type = random.choices(
        ["topper","average","coder","procrastinator","burnout"],
        weights=[15,40,20,15,10]
    )[0]

    # -------- DATA GENERATION --------

    if student_type == "topper":

        cgpa = round(random.uniform(9,9.9),2)
        study_hours = random.randint(5,8)
        coding_hours = random.randint(8,20)
        social_media = random.randint(0,2)
        sleep_hours = random.randint(6,8)

    elif student_type == "average":

        cgpa = round(random.uniform(7.5,8.5),2)
        study_hours = random.randint(2,5)
        coding_hours = random.randint(3,8)
        social_media = random.randint(2,4)
        sleep_hours = random.randint(6,8)

    elif student_type == "coder":

        cgpa = round(random.uniform(7,8.5),2)
        study_hours = random.randint(1,3)
        coding_hours = random.randint(12,25)
        social_media = random.randint(1,3)
        sleep_hours = random.randint(5,7)

    elif student_type == "procrastinator":

        cgpa = round(random.uniform(6.5,7.8),2)
        study_hours = random.randint(1,2)
        coding_hours = random.randint(1,4)
        social_media = random.randint(4,7)
        sleep_hours = random.randint(7,9)

    else:  # burnout

        cgpa = round(random.uniform(7,8.2),2)
        study_hours = random.randint(5,7)
        coding_hours = random.randint(4,8)
        social_media = random.randint(2,4)
        sleep_hours = random.randint(4,6)

    attendance = random.randint(60,95)

    # backlog probability
    if cgpa >= 9:
        backlog = "No"
    elif cgpa >= 8:
        backlog = random.choices(["No",1],weights=[80,20])[0]
    else:
        backlog = random.choices(["No",1,2],weights=[50,35,15])[0]

    values = [
        cgpa,
        attendance,
        backlog,
        study_hours,
        coding_hours,
        sleep_hours,
        social_media
    ]

    # -------- OUTLIERS --------

    if random.random() < OUTLIER_PROB:

        messy = [
            "around 8",
            "85%",
            "maybe 3",
            "depends",
            "idk",
            "none",
            "many",
            "few",
            "sometimes",
            "~7"
        ]

        values[random.randint(0,len(values)-1)] = random.choice(messy)

    # -------- FILL TEXT INPUTS --------

    inputs = driver.find_elements(By.XPATH,"//input[@type='text']")

    for field,value in zip(inputs,values):

        field.clear()

        for ch in str(value):
            field.send_keys(ch)
            time.sleep(random.uniform(0.02,0.08))

    # -------- RADIO QUESTIONS --------

    groups = driver.find_elements(By.XPATH,"//div[@role='radiogroup']")

    for g_index, group in enumerate(groups):

        options = group.find_elements(By.XPATH,".//div[@role='radio']")

        # Assignment completion
        if g_index == 0:

            if student_type == "procrastinator":
                choice = options[-1]
            else:
                choice = random.choices(options,weights=[40,40,18,2],k=1)[0]

        # Stress
        elif g_index == 1:

            if student_type == "burnout":
                choice = options[-1]
            else:
                choice = random.choice(options)

        # Motivation
        elif g_index == 2:

            if student_type == "topper":
                choice = options[-1]
            else:
                choice = random.choice(options)

        # Expected grade
        elif g_index == 3:

            if cgpa >= 9:
                choice = options[0]
            elif cgpa >= 8.5:
                choice = options[1]
            elif cgpa >= 7:
                choice = options[2]
            else:
                choice = options[-1]

        # Placement ready
        elif g_index == 4:

            if student_type == "coder":
                choice = options[0]
            else:
                choice = random.choice(options)

        # Exam preparation
        elif g_index == 5:

            if student_type == "topper":
                choice = options[0]

            elif student_type == "procrastinator":
                choice = options[-1]

            else:
                choice = random.choice(options)

        else:
            choice = random.choice(options)

        choice.click()
        time.sleep(random.uniform(0.3,0.8))

    # -------- SUBMIT --------

    driver.find_element(By.XPATH,"//span[text()='Submit']").click()

    time.sleep(random.uniform(2,4))

    try:
        driver.find_element(By.XPATH,"//a[contains(text(),'Submit another response')]").click()
        time.sleep(random.uniform(3,5))

    except:
        driver.get(FORM_URL)
        time.sleep(4)

driver.quit()