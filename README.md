# COMP-255-Program-6

Data:
Jane Austen novels in UTF-8 encoded form were sourced from Project Gutenberg
ChatGPT 4o works were generating by asking the chatbot to write "in thestyle 
of Jane Austen"
Lexos was used to create the tokenizer-table.csv document, which contains 
the proportion of the words in each work which were a specific word


Note on the usefulness of the data:
This project would be far more telling if the code were used with better 
data. ChatGPT4o was used to generate the AI texts and, despite persistent 
prompting, would not write very long stories. So the AI generated stories 
are considerably shorter than the stories written by Austen. This is not 
ideal: in the shorter space ChatGPT4o cannot write the same variety of words 
as Austen. In an attempt to combat this somewhat, the words in the csv file 
were required to appear in at least 8 documents (the number of Austen texts), 
rather than requiring words to appear in every document (and end up with very 
few colmumns as the AI texts are short) or accept every word in every document 
(and end up with way too many columns). The data was also standardized using
sklearn's StandardScalar before cluster analysis in attempt to combat these
issues.

Efficacy:
With this limited dataset, both cluster analysis and a SVM were able to 
separate the authentic Jane Austen texts from the ChatGPT 4o imitations.

