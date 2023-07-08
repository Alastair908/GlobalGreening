# GlobalGreening
This project aims to take a look at the Global Greening theory. Using a U-net CNN deep learning model to create an easily reproducible mask to focus on the wilder and less human influenced parts of the world and applying it to NDVIs (a measure of vegetation) taken during the months of May/June.

# Contents
1. Motive
2. Method
3. Results

## Motive
During the last 20 years, CO2 levels have risen from 370ppm to 440ppm. This seemingly minor increase has had demonstrable effects on our warming global climate. It stands to reason that plants would prefer an atmosphere containing more carbon for them to grow, hence the theory of 'Global Greening'.

Our project set out to identify whether or not increasing CO2 levels increase global vegetation growth. This idea has been investigated before but never on such a granular level. To measure the level of 'greening' caused by CO2, it was necessary to separate the different land uses that cover our planet. Only by doing so could we minimise the other human influences on vegetation volume and health such as deforestation, afforestation, eutrophication, intensive farming etc.

With this process we generated a 'mask' of the generally undisturbed natural land, perfect for seeing how vegetation has changed over the years. By placing this 'mask' over satellite data of density of vegetation across an area, today's results can be compared to the those of decades prior.

## Method
To train the model, we used Sentinal 2 satellite images and ESA world cover land use data from 2021 for the state of Colorado. Colorado was chosen as it contains a good mix of ecosystems, large cities and vast tracts of farmland. The system can then be used to predict the land use for other parts of the world based on the satellite images. The masks are then split into 'wild areas' and 'human influenced areas'.

The mask is then applied to May/June NDVI data for those predicted areas over multiple years. Using a starting point of 2002 the results can then be compared to see how levels of vegatation have changed during the greenest times of the year.

## Results
Our findings indicate a general increase in wild flora for the state of Colorado from 2002 to 2010. However, this growth has not been sustained through the decade with a clear drop in the density of vegetation post 2010. By 2022, the NDVI data shows a slightly higher rate of vegetation compared to 2002.

