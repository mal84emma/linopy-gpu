# Reflections on one-shotting cuOpt support

## Planning

Creating the planning docs and harness took about 2hrs

- The bulk of the plan came together very quickly, it was probably 80% within 45mins
- Grounding important details of the setup/implementation direction through actual checks of the machine and existing codebases (linopy & cuOpt) was very helpful
- Fable still makes some odd mistakes that need to be spotted through human review, e.g. details around how uv is invoked or assumptions from previous context that aren't validated
    - e.g. it assumed that the cuOpt DataModel was the best design choice despite that being a passing suggestion from an earlier turn (maybe it is, but baking in design decisions this early seems risky), and that cuOpt needed installing in a particular, slightly complicated way, that it really didn't
    - another e.g. some of the validation steps looked very sensible but actually didn't have any infrastructure to support the checking (e.g. saying that files would be checked that they were unchanged, but the files aren't git tracked)
- Once the plan gets large ensuring self-consistency becomes important - for some reason even Fable doesn't go and update all references in the plan when changes get made
    - when changing the authoring convention, several rounds of review and clean up were needed to make the doc consistent
- All this means that the plan still needs several rounds of review and refinement before it's ready, and likely still needs reading through carefully by hand with the context of the actual direction and tools
    - maybe some of the consistency issues that took lots of time were not super important, but the number of issues that are found in each review round are not confidence inspiring
    - I think this is particularly true of how tools are used within the project, often Claude will use standard conventions (e.g. for uv) that don't actually match the project setup

In conclusion: impressive ability to plan quickly, but the plan still needs careful review and direction with an understanding of what the actual goals are and the context of the project and its tooling.

For tasks with existing clear tests I can see this being easier and "all tests green" being enough of a direction, but where the code is being extended and it's not obvious what the tests should be, a good bit of thinking is needed about whether the plan is going in the right direction and whether the verification steps are actually signaling that the task is done.

## Implementation

Checking and correcting the implementation took about 4hrs (but by 1.5hrs it was probably good enough)

- Claude seems to have problems with monitoring progress of shell scripts sometimes (this goes back to the project specific tooling stuff I think), so maybe the implementation agent needs babysitting for a bit to determine some project specific rules/infrastructure for monitoring tests, scratch tries, experiment, etc. for the specific project
- A few of the design decisions made were a little questionable, e.g. the choice to not support certain features of either the cuOpt or linopy API because things "were never measured" or "there was no evidence", but the exploration phase of the project was exactly intended to do this measuring and gather evidence
    - maybe the agent system needs to be designed more flexibly so that at any stage issues can be pushed all the way back to evidence gathering and then escalated through the steps (planning, review, implementation), a bit more like what a human would do, but this makes coordination much harder
- Sometimes the documentation and reasoning the agents come up with are very opaque and required loads of context where there is often a simpler and clearer way of explaining the issues that just needs a bit of teasing out
    - making sure the documentation is human readable and clear enough to pick up later is important
- Some issues are just hard enough that they need human intervention to resolve, e.g. the OpenMP state leakage bug from repeated MIP solves by cuOpt
    - having a well defined testing/validation framework is very powerful, but it does limit the autonomy of the agents, as changing the tests is a design decision that you might not want to leave to another agent
- Session limits are a pain to manage when you're implementing autonomously
    - maybe you could set up some kind of hook to check whether the next turn should be delayed until the session limits reset, e.g. if usage > 90% delay the next turn until the next session starts
- Comments and docstrings are continuously overkill and full of Claude-isms
    - but a couple of examples of concise-ification and it can do quite a good job of making the comments more concise and readable
    - sooooo many long docstrings, particularly small functions where a single line and a function signature is enough to explain the function, but Claude will just write a 10 line docstring even for a 1-3 line trivial function
- Tests are generated for anything and everything, including behaviour of upstream packages that can't be controlled
    - maybe this is a good thing, but I can see it leading to loads of churn as package versions update. Lots of the things that are tested should just be noted as existing bugs
    - some of the tests are just unnecessarily complicated to test odd and irrelevant bits of behaviour, getting a clean and clear testing suite is pretty critical and Claude often pollutes the test files with fluffy nonsense
- The implementation is not exactly fast once you have all the planning and verification stage gates
    - I could probably get it done in fewer hours if I was going at it hard, but I couldn't do that overnight
    - the amount of my time saved is pretty significant. I reckon I spent about 6 hours on something that would probably take me more like 12+ doing it manually
- Going from the system's first completed draft (planned, implementation, iterated, reviewed) to a final version that is clean and high quality takes a decent amount of time and iteration
- If you keep asking Claude to find bugs it will keep finding them
    - maybe this is just a Python thing, really we should Oxidize
- With a thorough hand review I found a few quality tweaks that could be made, but nothing major

In conclusion: the plans and design approach needs a good bit of reviewing and iteration to achieve high quality, but pure implementation is pretty great, the only major issue is being inundated with docs fluff and tests that don't need to be there