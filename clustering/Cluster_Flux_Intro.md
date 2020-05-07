## What is the Zestimate and what is the logerror?

Before diving in to this whole 'Zestimate' thing, it may help to clear up a few ideas, at least as we understand them in pursuing our project's goal.

There are two main values dictating home sale prices:

1.) The Market Value = what the BUYER says the property is worth; and

2.) The Appraised Value = what the BANK says the property is worth.

Banks win 99% of the time because banks have 99% of the money, and that's a good thing - it helps mitigate reality between Buyer and Seller.

I can't demand my house sell for a million dollars when the bank says it's only worth a Happy Meal.  Conversely, the buyer can't buy my home for a Happy Meal when the bank says it's **AT LEAST** worth a Taco Bell Tripleupa Box.  (That took some practice.)

Undertanding this communication gap, Zillow was created in 2006 as a way of providing information to both home-buyers and home-sellers, the end goal being a mutual understanding at the beginning of price negotiations.  One of their flagship offerings is their 'Zestimate,' a constantly-updated and fine-tuned home valution model that is used to predict the market value of a home based on things like '*home facts, location, and market conditions*' (italics are directly from their website, https://www.zillow.com/zestimate/).

While strong and highly durable, the Zestimate is not perfect, even by it's own admission.  From the 'Median Error' section of the Zestimate website: "For most major markets, the Zestimate for on-market homes is within 10% of the final sale price more than 95% of the time."

Plain English: in cities of roughly a million or more people, the difference between Zillow's *predicted* home sale price is 10% different from the home's *actual* sale price.  Not bad, but on a \\$300,000 home, Zestimate can only ballpark a sales price range between \\$270- and \\$330-thousand dollars, a potential dream-crusher for both parties (but don't worry: banks still make out alright).

Because homes are not fiat currencies (they have actual, real value), Zillow can continually improve their model with tangible feedback in hopes of minimizing that error gap.

To see what may be driving this error, we are using what we learned in the Clustering Methodologies section of our Data Science Q-Course.  Instead of the listed 'Mean Error,' we are clustering to determine what is driving the 'logerror' experienced in Zillow's predictive model.  Using logerror (a column from our provided MySQL database) means that we are assuming a distribution underlying Zillow estimates and actual home sale prices. 

### NB:

You may be wondering why we're dealing with California data.  At least we know we were.

Turns out, Texas (and a handful of others) is a non-disclosure state, and the Texas Real Estate Commission - Rulers over all things Texas Real Estate - is under no legal obligation to provide any home sale price information to outside companies like Zillow or RedFin (the Pepsi-cousin to Zillow's Coke). 

Take that for what it's worth, but that leads us to believe the logerror drivers we discover will be unique to the California Zestimate model, and cannot be applied universally without a sacrifice in overall accuracy.


