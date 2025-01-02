__Math Review__:
![image](https://github.com/user-attachments/assets/d989fbfc-3387-4037-9ea7-9fa4c5b3ef18)
![image](https://github.com/user-attachments/assets/ec8b5d5a-b9ee-4580-b6c8-ffd5d7ced7a6)

__ChatGPT__:

After reading through the rest of the assignment, I saw how it was recommended to use np.random.choice but I was not sure how to implement that function of what it did. I looked it up into ChatGPT and it helped me better understand the parameters and the return value.

__Part A:__

By Class:

![hist1](https://github.com/user-attachments/assets/d557bd6a-bf8b-433c-9362-9eabd5154c80)

Survived = 0 had a frequency over 250 for class 3, which is much higher than the frequencies of Survived = 0 for the other classes because those frequencies never exceeded 100. Assuming Survived = 0 means that the passenger did not survive, we can assume that class 3 was the lowest passenger class on the ship. This would make class 1 the highest class, which would make sense given the supporting data and information we know. For class 1, people were more likely to survive the wreck than not survive. This was the only class where this was the case. Class 2 was a lot closer with both frequencies being around 80.

By Sex:

![hist2](https://github.com/user-attachments/assets/7d86f40a-dbae-4b62-90a1-ae653f8111c6)

Like the class graph, this graph is a bit ambiguous due to the weird format and lack of explanatory labels. Given the information that was given in the problem set, we can assume that Sex = 0 were females because this group had a much higher chance of surviving than not surviving. On the contrary, the other sex had a significantly higher chance of not surviving than surviving.

By Age:

![hist3](https://github.com/user-attachments/assets/a5b84638-ed44-4f88-855e-3658040a1c80)

This graph is a lot easier to read and understand. Again, the legend uses two variables of the same name which can make things confusing, but I assume that we are supposed to be looking at the Boolean value each variable is assigned to. This graph tells us that most of the passengers were between the ages 20-40. Each age group had a higher likelihood of not surviving than surviving except for the youngest one. This is likely because the passengers on the ship made it a priority to help the youngest passengers first. The oldest age group seems to have had the same frequency of death to survival. This graph puts into perspective how devastating this shipwreck was.

By number of siblings and spouses of the passenger:

![hist4](https://github.com/user-attachments/assets/04ce1a79-b7b5-4fee-a7de-511e67e2d3db)

This graph was confusing, and I had to look up the dataset online to get a better understanding of what it was showing. I am having a hard time making any assumptions based off this graph. Every group had a higher frequency of death than not except for the group where the passengers only had one sibling and spouse aboard. The difference is not too large so no real conclusions or statements can be made about this graph.

By number of parents and children of the passenger:

![hist5](https://github.com/user-attachments/assets/e401c6d8-2344-46c1-be4b-59620b752cc9)

Like the previous graph, most of the passengers fell into the category of 0. Unlike the other graph, both 1, 2, and 3 show the passengers having a higher frequency of surviving than not surviving. Like the other graph, I am not sure what real conclusions can be made about this graph. Personally, I think other graphs we have seen have been more telling.

By Fare:

![hist6](https://github.com/user-attachments/assets/e83d525a-9a80-49d9-ba3e-6f6cb697f17a)

This graph shows that as the fare price increases, the chance of survival also seems to increase. The passengers in the first group for fare price were twice as likely to die than survive. Every other fare price group had a higher chance of survival than death.

By Embarked:

![hist7](https://github.com/user-attachments/assets/5a8583c4-7966-43af-bc74-c2d51581136f)

This graph is not descriptive. Those that embarked from 0 survived more than they did die. Not many passengers embarked at 1. A majority of the passengers embarked at 2 and as expected, the frequency of deaths was higher than the frequency of survival.

__Scatterplot__:

![scatterplot](https://github.com/user-attachments/assets/b0a69b19-58e3-49e2-926c-8ce6ecfdb4a1)

There are two passengers who were outliers in terms of fare and they survived. It appears as if passengers who were younger survived more than passengers who were older but more information is needed to make complete conclusions and a better graph with less overlapping points can paint a better picture.

__Relection Questions__:
1.	What is your GitHub username?
lilscolari
2.	Number of hours spent working on this problem set:
5-6 hours.
3.	What you hope to get out of this course:
I hope to learn new machine learning techniques that I can use to apply to datasets that I find fascinating. I have experience in data science and want to see how different techniques compare to each other and gain a better understand of when to use certain techniques.
4.	In 1-2 sentences, please describe a positive change you would like to see in the world, and how Machine Learning could help you to achieve that goal:
I want to see food not being wasted and using machine learning techniques, we can prevent food waste.
5.	Your previous experience (if any) in machine learning and/or deep learning:
I have done a research project using linear regression. I took a class and briefly learned about k-nearest neighbors.
6.	 (Optional) Feel free to let me know what you liked/disliked about this homework, what you learned, etc:
I liked the homework assignment but sometimes I felt lost and had to reread the pdf. I would prefer it if the instructions were also in the README file.
