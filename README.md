# hello-world
the first respository

Hello, this is the first file of xiaojitui.
test, test, test

## I am based in Nashville. I just joined Liberty in November, so it’s my first month, I previously worked on GenAI projects at a finance company, and I’m glad to bring that experience to the team and contribute to our AI work.

## One of the AI projects I’ve been working on is the Lex Bot. Just a quick recap: Liberty launched the new NLU-enhanced Lex Bot in October. This upgrade was designed to improve intent recognition compared with the original bot. However, the initial A/B testing showed a slightly lower containment and a higher re-segmentation rate, so our data science team started investigating.

## What we did:
First, we developed a GPT-assisted evaluation approach to assess the NLU bot’s performance quickly and at scale.
Second, we connected the evaluation results to the business impact, specifically re-segmentation.
And finally, we worked closely with Zach and Sean from the Claims team to review and validate the findings.

## What we found:
From a data science perspective, we identified the specific intents where the NLU bot is underperforming, which gave us a clear direction for model improvement.
From an IVR design perspective, working with the Claims team, we found two main issues:
1.	IVR Prompts are too generic, leading to unclear caller responses. For example, the bot often asks, “Describe what happened in a few words,” and callers just answer like “I have an accident” or “I hit someone.” These generic statements make it difficult for the NLU-bot to detect the actual intents of the clients.
2.	Mapping from Intents to loss-causes is not clearly defined, so even when an intent is recognized by the bot, it may not route correctly, leading to re-segmentation.
Zach & Sean, do you want to provide more comments on these two issues? In addition, we also noted some transcription errors that need to be addressed.

## Next steps:
On the data science side, we’ve finalized the GPT evaluation process to speed up issue detection.
Also, in partnership with the Claims team, we’re working to improve two things, first, the mapping between intents and loss causes, and second, and the IVR design such as adding more specific questions and follow-ups to help callers provide clearer information. 
Finally, we’re collaborating with the tech team to implement these recommendations into production and explore potential model improvements. 
