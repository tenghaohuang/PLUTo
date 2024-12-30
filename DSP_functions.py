import dspy
# ## Define components of our DSP program

# In[106]:
class AugmentQuery(dspy.Signature):
    """
    Given the user query below, identify and list the potential intents the user might have. The identified intents should not only capture explicit and implicit actions or requests but also reflect the general topic domain or subject matter of the query.
    """
    user_query = dspy.InputField(desc="Ultimate goal that we want to solve.")
    user_intents = dspy.OutputField(desc="User intents.")

# class PredictNextAction(dspy.Signature):
#     """Divide the user request into atomic actions.
#     Then, based on the current information, predict a next atomic action to take in order to achieve the user request.
#     Make sure the output of the next atomic action does not belong to one of the current information.
#     Always answer only the predicted next atomic action in a noun phrase.
#     If the user request is already achieved, return one word, 'STOP' """

#     current_information = dspy.InputField(desc="Information that we currently have.")
#     user_request = dspy.InputField(desc="Ultimate goal that we want to solve.")
#     next_action = dspy.OutputField(desc="Next action to take.")

class PredictNextAction(dspy.Signature):
    """Predict a simple next action that will help solve a user request. Make sure to break down the user request into smaller parts while predicting the simple next action. Do not repeat previous actions. If the user request is already achieved, return one word, 'STOP' """

    current_information = dspy.InputField(desc="Information that we currently have")
    previous_actions = dspy.InputField(desc="Previous actions taken")
    user_request = dspy.InputField(desc="Ultimate goal that we want to solve.")
    next_action = dspy.OutputField(desc="Simple next action in noun phrase")


# In[107]:

examplar = """
Usecases:
1. 'Research reputable sources on the history of the Civil Rights movement in the United States.',
2. 'Provide a list of recommended family-friendly resorts in Hawaii and suggest activities to do with kids.',
3. "Provide the user with the list of academic papers and articles discussing criticisms and controversies around Jean-Luc Godard's works.",
4. "Research Akira Kurosawa's filmography and find reliable sources for critical analysis and reviews of his movies.",
5. 'Provide background information on the Western film genre and recommend famous films.'
------------------------------------------------------
API description: search - <search>' API for concise summaries on various topics from trusted sources, catering to diverse research needs.
score: 0.022435897435897433

API description: <search> enables users to quickly find relevant information on a specific topic by searching Wikipedia articles. It allows users to explore related entities, expand their knowledge, and discover new topics of interest.
score: 0.04807692307692308

API description: The 'search - <search>' API is your gateway to knowledge, capable of searching detailed summaries from sources on historical events, suggesting travel destinations, and providing analyses on cinematic works. It helps seeking background information, curated lists, or academic insights, this API facilitates a diverse range of research inquiries.
score: 0.14423076923076922
------------------------------------------------------
"""

class OGPredictAPI(dspy.Signature):
    #"""Please predict the next API to execute in order to solve the user query step by step. You are given candidate APIs to choose from."""
    #"""Predict the next API to execute in order to solve the user request. You are given candidate APIs to choose from."""
    """Select one relevant and useful API to execute from the given candidate APIs in order to achieve the current goal. The selected API and user query should be similar in domain and subjects.
    The output must be one word. The selected API must be one of the candidate_apis. DO NOT hallunicate. If there is no relevant API, output NONE."""

    candidate_apis = dspy.InputField(desc="Candidate APIs that can be used.")
    current_goal = dspy.InputField(desc="Current goal we want to solve.")
    predicted_api = dspy.OutputField(desc="A selected API.")


# In[108]:


class ExecuteAPI(dspy.Signature):
    """Imagine we have executed the API using the information we currently have in order to achieve the given goal. Predict the output of the execution. Always answer in noun phrases."""

    executed_api = dspy.InputField(desc="API that we have executed.")
    current_information = dspy.InputField(desc="Current information we have.")
    goal = dspy.InputField(desc="goal we want to achienve by executing the api")
    output = dspy.OutputField(desc="Output after executing the API.")

class PredictSimilarAPI(dspy.Signature):
    """Given the executed API, predict similar APIs that can be used to achieve the same goal. The output must be a list of APIs. If there is no similar API, output NONE."""

    executed_api = dspy.InputField(desc="API that we have executed.")
    candidate_apis = dspy.InputField(desc="Candidate APIs that can be used.")
    relevant_apis = dspy.OutputField(desc="Relevant APIs to the source API.")

class GTSPredictAPI(dspy.Signature):
    """Select all relevant and useful API to execute from the given candidate APIs in order to achieve the current goal. The selected API and user query should be similar in domain and subjects.
        The output must be only API names seperated by ','. The selected APIs must be one of the candidate_apis. DO NOT hallunicate. If there is no relevant API, output NONE."""

    candidate_apis = dspy.InputField(desc="Candidate APIs that can be used.")
    current_goal = dspy.InputField(desc="Current goal we want to solve.")
    predicted_apis = dspy.OutputField(desc="A selected API.")

class DescriptionOptimization(dspy.Signature):
    """
    The examplar contains some api descriptions along with their corresponding scores. The descriptions are arranged in ascending order
    based on their scores, where higher scores indicate better quality. Now given a set of API use cases,
    rewrite source API description to make it more relevant to the query. The API name should be consistent to the provided API.
    The new API description should be different from the old descriptions.
    """
    examplar = dspy.InputField(desc="Examplar of the optimization task.")
    use_cases = dspy.InputField(desc="Several use cases of the API")
    source_api_description = dspy.InputField(desc="Source API Description (To be optimized).")
    optimization_log = dspy.InputField(desc="Log history of optimization")
    new_api_description = dspy.OutputField(desc="Optimized API Description")

class DescriptionOptimization_new(dspy.Signature):
    """
    Given the use cases and reasons provided below, write a comprehensive and detailed description for the API, highlighting its capabilities.
    """
    use_cases = dspy.InputField(desc="Several usecases and helpful reasons of the API")
    new_api_description = dspy.OutputField(desc="Optimized API Description")
class predict_api_useful_reason(dspy.Signature):
    """
    Given a user query and an API, predict a reason why the API would be helpful to achieve the user query.

    Here is an example:
    user_query:  Can you search for websites and literature on the impact of social media on mental health? Please compile a summary of the major arguments and evidence.
    API: create_file - Create a pptx file with specific theme, available thems  green / orange / tech / wooden / flat.
    useful_reason: The "create_file" API allows for the creation of themed PowerPoint presentations. In addressing the user's query, it can transform findings on the impact of social media on mental health into a visually appealing and structured format, making the information easy to understand and share.
    """
    query = dspy.InputField(desc="The user query")
    API = dspy.InputField(desc="The helpful API to address user query and its description")
    useful_reason = dspy.OutputField(desc="Reason that the API would be helpful")


class FirstPlanning(dspy.Signature):
    """
    You are given a user request requiring multi-step actions and reasoning. Please tell me what action to take first accurately in a noun phrase.
    """
    user_request = dspy.InputField(desc="User request that may require multi-step actions.")
    next_action = dspy.OutputField(desc="Simple next action to take in noun phrase.")

class LatterPlanning(dspy.Signature):
    """
    You are given a user request requiring multi-step actions and reasoning. Please tell me what action to take next in a noun phrase. If the user request is already achieved using the previous actions, return 'STOP'.
    """
    previous_actions = dspy.InputField(desc="Previous actions you took to achieve the user request.")
    user_request = dspy.InputField(desc="User request that may require multi-step actions.")
    next_action = dspy.OutputField(desc="Simple next action to take in noun phrase.")

class PredictAPIs(dspy.Signature):
    """For each given candidate APIs, give a reasoning how the API would be helpful to achieve the current goal. Then, return a list of most relevant APIs that would help achieve the current goal. The output must be only API names seperated by ','. DO NOT hallunicate. If there is no relevant API, output 'NONE'."""
    #Select one relevant and useful API to execute from the given candidate APIs in order to achieve the current goal. The output must be one word. The selected API must be one of the candidate_apis. DO NOT hallunicate. If there is no relevant API, output 'NONE'."""

    candidate_apis = dspy.InputField(desc="Candidate APIs that can be used.")
    current_goal = dspy.InputField(desc="Current goal we want to solve using APIs.")
    predicted_apis = dspy.OutputField()

##################Query Anonymization#####################
class QueryAnonymization(dspy.Signature):
    """
    Prompt:
    Given an input query, remove the specific entity information and return only the general query template. Make sure your method is generalizable to handle other queries.

    Demonstration:
    Input_query: "Gather information about blockchain technology."
    Output_query: "Gather information about..."

    Input_query: "Provide literature on the benefits and drawbacks of Agile and Waterfall project management methodologies and summarize the findings.
    Output_query: "Provide literature on ... and summarize ..."

    Input_query: Research different regions in Japan to identify off-the-beaten-path destinations.
    Output_query: Research ...

    Input_query: Compile the research findings into an organized outline for the PowerPoint presentation.
    Output_query: Organized outline for the PowerPoint presentation
    """
    Input_query = dspy.InputField(desc="Input query containing specific entity information.")
    Output_query = dspy.OutputField(desc="Output query containing only general query template.")
    
class PredictAPIs_OneShot(dspy.Signature):
    """For each given candidate APIs, give a reasoning how the API would be helpful to achieve the user query. Then, return a list of most relevant APIs that would help achieve the current goal. The output must be only API names seperated by ','. DO NOT hallunicate. If there is no relevant API, output 'NONE'."""
    #Select one relevant and useful API to execute from the given candidate APIs in order to achieve the current goal. The output must be one word. The selected API must be one of the candidate_apis. DO NOT hallunicate. If there is no relevant API, output 'NONE'."""

    candidate_apis = dspy.InputField(desc="Candidate APIs that can be used.")
    user_query = dspy.InputField()
    predicted_apis = dspy.OutputField()