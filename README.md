## Note: This project is still under development and is not completely accurate.

# Stacks Call Number
Small personal project that detects call numbers from library book images and checks their order. 

## Here is a quick breakthrough of the project:

1. A ```yolov8``` model trained to detect call numbers from the books is used to detect the call numbers.

![image](https://github.com/rosnMagar/stacks_call_number/assets/146673489/fff446c4-1beb-4dcb-8a50-5579a8576185)

1. Each of the call numbers are extracted and using ```easyOCR``` Optical Character Recognition is performed

![image](https://github.com/rosnMagar/stacks_call_number/assets/146673489/8f7b3f93-12db-4084-a61b-1d2da5b6b771)

1. Finally the call numbers are parsed using ```pycallnumbers``` and ordered

![image](https://github.com/rosnMagar/stacks_call_number/assets/146673489/0144bf7b-d67a-40a2-9584-42b8de2565be)

**Numbers:** The predicted order in which the books should be placed.

**Yellow Rectangle:** The call numbers were probably read incorrectly by the OCR engine.

**Green Rectangle:** The call numbers were probably read correctly by the OCR engine.

There are two different modes to predict the order: Ignore the flagged(yellow rectangles) or do not ignore the flagged rectangles.
