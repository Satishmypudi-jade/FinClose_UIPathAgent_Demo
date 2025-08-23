# GenBoxes

Known Stakeholders: Jade, Debasish

Purpose: To provide an application that shows users the current state of instances of one of their business processes including the ease of seeing ones that are at risk of not meeting their SLA. This is done by providing an app with a dashboard to display that information, the connections to a database where that information can be fed to, and a way to be able to set up any business process desired. We have also explored a page to show duplicates and to make the business process with AI. The current implementation is focused on providing the AI ability with the familiar chatbot set up in the bottom corner (rather than its own page).

Current Focus: To build out a multi channel design for the original Streamlit GenBoxes project. Actually, more to have a stand alone streamlit. This will allow for different deployments. All code that must vary, however, would have to be incapsulated within a multi channel directory. 

# Structure
.streamlit - a folder with a single config file for some Streamlit settings

architectural_diagram_items - a folder with images that can be convenient for creating an architectural diagram for the solution

assets - a folder for images (maybe could have other non textual stuff if that was necessary...)

data - a folder containing the table information at a certain point in time not high priority...

discoveries - a folder for cool insights about what can be done with Streamlit that could be pulled into other things... meant mostly as a spot to put things to move to another repository

environments - this part of the solution is meant for breaking down meaningful distinctions of different environmental implementations. Generic is meant for providing objects that don't rely on a specific environment. These are often replaced with specific implementations in the other environments. There is also a grouping for AI and SQL. Please refer to the file environment.selection to see how files can be pointed as desired to the correct elements.
environments/local - This is for both streamlit cloud (outside of Snowflake) and local. They could be separated for differences in allowing for dotenv, but that hasn't been necessary.

pages - this holds the code that doesn't have to very between different environments and isn't the same for different pages, but rather, the code that is specific to each page

programming_scripts - little python scripts that could be used for helping out at the time of writing this none of them are being used and the one in there is just one to get all of the objects defined in a model (to then be able to import specific items rather than *) but that may not ultimately be used for anything

tests - testing stuff

environment_selection.py - the place to pick what to import from different environment options; it also merges together elements from environments (e.g. to pull together local file storage and to snowflake generative ai)

main.py - the main app

requirements.txt - just a list of the needed package installations for streamlit cloud (not the full list)

testing.py - a file to run chatbot tests for the different categorizations of messages (not put in a folder because of importing difficulties...that may just be dumb but that's why)

# Considerations
This section will walk through considerations that went into the specific design that is given. Each one is considered in light of something that would have seemed to be easier and more elegant.

Consideration #1: Why not have the file names presented like 1_🏠_Dashboard.py rather than 1_(house)_Dashboard.py? 

Experiment:
code in a streamlit in snowflake instance: import streamlit as st
st.write("hi") # This is probably unnecessary for the experiment...
st.switch_page("pages/1_🏠_Dashboard.py") # to be contrasted with st.switch_page("pages/1_Dashboard.py") which work

With the packages:
Python (can't change)
Streamlit (Latest, 1.39.0, and 1.35.0) Varied because I have found significant differences between 1.35.0 before and Latest (at least before 1.39.0 existed as an option)
snowflake-snowpark-python (Latest and 1.24.0 I believe lastest as of 1/29/2025)

Error Message:
Error running Streamlit: [SERVER_ERROR] Python Interpreter Error: TypeError: bad argument type for built-in operation

Conclusion:
It is reasonable to believe that Snowflake does not support non-ascii characters within the switch_page function, therefore, any such items must be dealt with in a less straightforward manner when working within Streamlit in Snowflake. Page link may not have the same restrictions but I haven't figured out how to make a menu out of those items....

Best Workaround Considerations:
There are two possible options with the workaround's application 1) apply it to all channels for better consistency between channels, or 2) apply it to just the affected channel. I would support 1 because having a single instantiation of an item allows for easier switching between channels especially unless something can be truly left alone without any changes. Then, for more specifics on the implementation, I could 1) name pages in a way that ascii characters are used for the non ascii characters and then swapped out for the presentation (similar to using numbers for ordering which is used in another project but not displayed as such) or 2) avoid including them in the names at all. I go with 1 because it is the easiest way to support something dynamic. Everything for the page name display is included in its name just not in the form it is finally presented in.

Consideration #2: Why are you overriding things after importing them?

Reasoning: It's easier.

Okay, I am willing to accept that this *could* be something that isn't a best practice, but it doesn't seem like it to me. More specifically, I know there is something out there about not overwriting functions. However, to my knowledge, the issue would be changing code from others that should be changed. The functions in this case would not be doing that. Instead, they are making it so that there is a default that would be used unless a better version overrides it. The biggest risk seems to be that it might be harder to trace if the override function broke and the first silently filled in... to a degree.

Conclusion: This system works smoothly for our needs. If something different were to be done, it could be implemented later to avoid as much effort for maintaining what exacly should be imported and what shouldn't (because a new version will be created).

Consideration #3: Why have a whole folder system for each possible environment plus generic plus generic for types of environments, etc? 

Reasoning: Having folders for any overlapping environments allows for one to avoid repeating oneself...although it is risking having a super complex import environment in which imports are happening to import what was imported before...Additionally, this could lead to problems if changes are made to generic elements without testing in all affected environments. 

However, there are multiple advantages to a single point of change where it is possible. For example, its a single point of improvement in most instances. It also as mentioned avoids repetition, where repetition and a web of imports are fundamentally about the same issue. This at least tries to organize it and make it simpler....

Furthermore, the items can be flagged in generic to know that there are overrides elsewhere and just looking up def <object_name> could pull up the other instances that would need to be updated (while one knows that they are different for a reason...).

Consideration #4: What if we put the selection into the settings...we would have to have the options but then you could....

No, this is bad, we need to be able to exclude libraries completely for environments to not have issues with unnecessary imports for those environments.

Consideration #5: How can 0 be converted to false .env?

# To Do List
- Create a table of textual information to describe the data available...
        - sqlite
        - snowflake
- Fix chat popover window display (challenging-low priority because we have another page for it; will I ever figure it out.... to be seen)


