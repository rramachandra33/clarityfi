"""Labeled routing examples: (question, expected intents, advice_request).

`expected intents` is the set of agents that SHOULD run. A None intent set means "ambiguous: the heuristic
may defer to the LLM" (confident == False is acceptable, and the fallback must still be sensible).
"""

F, P, M, G, N, T = "finance_qa", "portfolio", "market", "goal_planning", "news", "tax"

CLEAR = [
    # ---- general education -> finance_qa
    ("What is an ETF?", {F}, False), ("Explain how compound interest works", {F}, False),
    ("What's the difference between stocks and bonds?", {F}, False), ("How does diversification reduce risk?", {F}, False),
    ("Define expense ratio", {F}, False), ("Why is inflation bad for savings?", {F}, False),
    ("Tell me about index funds", {F}, False), ("What is dollar-cost averaging?", {F}, False),
    ("How do mutual funds work?", {F}, False), ("What is a bond yield?", {F}, False),
    ("Can you explain what a P/E ratio is?", {F}, False), ("What is the difference between AAPL stock and an S&P 500 ETF?", {F}, False),
    ("How does a 60/40 portfolio work?", {F}, False), ("what is an emergency fund", {F}, False),
    ("Explain risk tolerance in simple terms", {F}, False), ("What is a stock split?", {F}, False),
    # ---- portfolio
    ("How is my portfolio doing?", {P}, False), ("Is my portfolio diversified enough?", {P}, False),
    ("Analyze my holdings", {P}, False), ("How risky is my portfolio?", {P}, False),
    ("What do my fees look like?", {P}, False), ("Am I too concentrated in one stock?", {P}, False),
    ("Review my asset allocation", {P}, False), ("What is the expense ratio of my funds?", {P}, False),
    ("How diversified am I?", {P}, False), ("Check my portfolio for concentration risk", {P}, False),
    ("What's in my investments?", {P}, False), ("Evaluate my portfolio", {P}, False),
    # ---- market
    ("What is AAPL trading at?", {M}, False), ("How is Tesla doing today?", {M}, False),
    ("Price of bitcoin", {M}, False), ("How has the S&P 500 performed this year?", {M}, False),
    ("Show me MSFT and GOOGL prices", {M}, False), ("What's VTI at right now?", {M}, False),
    ("Is the market up today?", {M}, False), ("NVDA stock price", {M}, False),
    ("Give me the trend for QQQ", {M}, False), ("How volatile is BTC-USD?", {M}, False),
    ("how are Apple and Microsoft doing", {M}, False), ("current price of $TSLA", {M}, False),
    # ---- goal planning
    ("Am I on track for my retirement goal?", {G}, False), ("How much do I need to save for a down payment?", {G}, False),
    ("I want $50,000 in 5 years, can I do it saving $300 a month?", {G}, False), ("Can I retire at 60?", {G}, False),
    ("How long will it take to reach $100,000?", {G}, False), ("What if I save $500 more per month?", {G}, False),
    ("I'm saving for a house, how much should I put away monthly?", {G}, False), ("Project my savings goal", {G}, False),
    ("When can I retire if I save $2,000 a month?", {G}, False), ("How fast can I reach $20,000?", {G}, False),
    # ---- tax
    ("What is a Roth IRA?", {T}, False), ("How are capital gains taxed?", {T}, False),
    ("Traditional IRA vs Roth IRA", {T}, False), ("What are the 401k contribution limits?", {T}, False),
    ("Explain tax-loss harvesting", {T}, False), ("What is an HSA?", {T}, False),
    ("How does a wash sale work?", {T}, False), ("Are dividends taxable?", {T}, False),
    ("What is a required minimum distribution?", {T}, False), ("How do 529 plans work?", {T}, False),
    # ---- news
    ("What's the latest news on Tesla?", {N}, False), ("Any headlines about the Fed today?", {N}, False),
    ("News about Apple earnings", {N}, False), ("What's happening in the markets this week?", {N}, False),
    ("Why is NVDA down today?", {N, M}, False), ("Latest on interest rates", {N}, False),
    # ---- multi-intent
    ("How is my portfolio doing and what is AAPL trading at?", {P, M}, False),
    ("Is my portfolio diversified, and am I on track for retirement?", {P, G}, False),
    ("What are the tax rules for selling VTI, and what's its price?", {T, M}, False),
    ("Explain what an ETF is and analyze my portfolio", {F, P}, False),
    ("What's the news on Microsoft and how is MSFT stock doing?", {N, M}, False),
]

ADVICE = [
    ("Should I buy TSLA?", {M}, True), ("Should I sell my bonds?", None, True), ("What should I invest in?", None, True),
    ("Is Apple a good buy right now?", {M}, True), ("Which ETF should I buy?", None, True),
    ("Tell me what to buy", None, True), ("Recommend some stocks for me", None, True),
    ("Should I put my savings in crypto?", None, True), ("Where should I invest $10,000?", None, True),
    ("Will NVDA go up next week?", {M}, True), ("When should I sell my shares?", None, True),
    ("What are the best stocks to buy now?", None, True), ("Is bitcoin a smart investment?", {M}, True),
    ("Should I move my 401k into bonds?", {T}, True), ("How should I invest my bonus?", None, True),
]

NOT_ADVICE = [
    "What is a stock?", "How do I read a bond quote?", "Explain what buying on margin means", "Why do people sell short?",
    "What does a financial advisor do?", "What is the difference between a buy and a hold rating?",
    "How does the stock market work?", "What happened to the market in 2008?",
]

# Deliberately ambiguous / awkward: heuristic must not crash; any of the allowed intents is acceptable.
AMBIGUOUS = [
    ("thoughts?", {F}), ("hi", {F}), ("can you help me with money stuff", {F}), ("what about bonds", {F}),
    ("is it a good time", {F, M}), ("tell me more", {F}),
]


# Two held-out sets written AFTER the rules were tuned (each was run once before any fixes; the misses
# they exposed were then fixed and they now serve as regression tests). Rule: when the heuristic stage is
# CONFIDENT it must be right; unsure questions are the LLM stage's job.
HELD_OUT_1 = [
    ("Could you walk me through how bonds pay interest?", {F}, False), ("Why do index funds usually beat active managers?", {F}, False),
    ("What does it mean when a fund is 'actively managed'?", {F}, False), ("I keep hearing about rebalancing, what is it?", {F}, False),
    ("How much of my paycheck should go to an emergency fund?", {F}, False), ("what is a dividend yield", {F}, False),
    ("Is my allocation too heavy in stocks?", {P}, False), ("Tell me whether my investments are balanced", {P}, False),
    ("What's my exposure to bonds?", {P}, False), ("How much am I paying in fund fees each year?", {P}, False),
    ("Give me a health check on my holdings", {P}, False), ("Do I own too much of a single company?", {P}, False),
    ("Where is Amazon trading?", {M}, False), ("How did Nvidia do over the last three months?", {M}, False),
    ("what's ETH worth right now", {M}, False), ("Quote for SPY please", {M}, False),
    ("How has gold done recently?", {M}, False), ("Compare the prices of VTI and VOO", {M}, False),
    ("If I put away $400 a month, will I have $30,000 in six years?", {G}, False), ("I need $15,000 for a wedding in 2 years, is that realistic?", {G}, False),
    ("How much should I be saving each month to retire comfortably?", {G}, False), ("Am I saving enough for my kid's college?", {G}, False),
    ("Will I hit my target if I keep contributing $250 monthly?", {G}, False),
    ("Do I pay tax when I sell an index fund?", {T}, False), ("What's the deal with Roth conversions?", {T}, False),
    ("How are qualified dividends treated at tax time?", {T}, False), ("Can I deduct contributions to a traditional IRA?", {T}, False),
    ("Any recent news about Amazon?", {N}, False),
    ("Is Tesla worth buying?", {M}, True), ("Would you pick Vanguard or Fidelity for me?", None, True),
    ("Give me a stock tip", None, True), ("What's the smartest thing to do with $5,000?", None, True),
    ("Break down my portfolio and tell me what the S&P 500 is at", {P, M}, False),
    ("Explain inflation, then check whether I'm on track for retirement", {F, G}, False),
    ("Hello!", {F}, False), ("What's a good book about investing?", {F}, False),
    ("How does the stock market work and what is AAPL at?", {F, M}, False), ("Is now a good time to buy Microsoft?", {M}, True),
]
HELD_OUT_2 = [
    ("Why do bond prices fall when rates rise?", {F}, False), ("What's the point of owning international stocks?", {F}, False),
    ("How do target-date funds work?", {F}, False), ("What is a covered call?", {F}, False), ("Explain the rule of 72", {F}, False),
    ("Are my funds too similar to each other?", {P}, False), ("Which of my holdings is the riskiest?", {P}, False),
    ("Tell me about the balance between stocks and bonds in my accounts", {P}, False), ("Does my portfolio have too much tech?", {P}, False),
    ("How many dollars do I have invested in total?", {P}, False),
    ("Show me how Google's stock has moved this quarter", {M}, False), ("What are the current prices of BND and AGG?", {M}, False),
    ("Is Ethereum up or down today?", {M}, False), ("What's the Dow at?", {M}, False), ("how's the nasdaq been lately", {M}, False),
    ("I'd like to have $200,000 by age 55, what monthly amount gets me there?", {G}, False), ("Will $500 a month get me to a million?", {G}, False),
    ("How many years until I can afford a $40,000 car if I save $600 monthly?", {G}, False), ("Are my savings on pace for buying a home in 2030?", {G}, False),
    ("Do I owe tax on interest from a savings account?", {T}, False), ("How does the IRS treat crypto gains?", {T}, False),
    ("What's the penalty for early 401k withdrawals?", {T}, False), ("Difference between a Roth 401k and a Roth IRA", {T}, False),
    ("What's the buzz around Intel this week?", {N}, False), ("Any updates on the Fed meeting?", {N}, False),
    ("Should I buy the dip in Nvidia?", {M}, True), ("Do you think Amazon will rise next month?", {M}, True),
    ("What is my risk level, and explain what beta means", {P, F}, False), ("Check my diversification and show the price of VTI", {P, M}, False),
    ("Am I on track for retirement, and what are the tax rules for IRAs?", {G, T}, False), ("Define inflation and tell me what BTC is worth", {F, M}, False),
    ("Can you summarize how investing works and what I should read first?", {F}, False), ("Analyze my portfolio and my goal progress", {P, G}, False),
]


# Found in live use: a ticker inside a TAX or NEWS question is not a price request.
CLEAR += [
    ("How much tax will I owe if I sell my VTI shares?", {T}, False), ("Are dividends from VTI taxable?", {T}, False),
    ("Do I owe tax when I sell TSLA?", {T}, False), ("What's the tax treatment of AAPL dividends?", {T}, False),
    ("What's the tax rule for selling VTI, and what's its price?", {T, M}, False),  # an explicit price ask keeps the market agent
]
