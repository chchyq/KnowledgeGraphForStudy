# Structure

In this project

- [YaleCraw.py](http://YaleCraw.py) is the file used in crawling the data from the open source classes of Yale, whose results are generated into course1.csv and folder course1.
- folder course1 is the folder including all the lectures of classes. Each .txt represents a class
- course1.csv includes information for all classes
    
    ```python
    Department = []  
    Course_Title = []  
    ListUrl = []  # the sublink of classes
    Course_Number = []  # the number of classes
    About_the_Course = []  
    Course_Structure = []  #the structure of classes
    Professor = []  # the professor of classes
    Description = []  # Description of classes
    Texts = [] 
    Lectures=[]
    ```
    
- [wikifier.py](http://wikifier.py/) is the file using wikifier tool to extract annotation and pageRank depending on the folder course1. Results are generated in folder wikifier1