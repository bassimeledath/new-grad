
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt 
import seaborn as sns
import math

def main():

    plt.style.use('fivethirtyeight')

    st.image('new_grad.jpg', use_column_width=True)

    @st.cache
    def load_data(nrows):
        data = pd.read_csv('with_studio.csv', nrows=nrows)
        return data

    @st.cache
    def load_data2(nrows):
        data = pd.read_csv('big_df.csv')
        return data

    grouped_df = load_data(400)
    big_df = load_data2(62615)

    st.header('Data at a Glance')

    wage_to_filter = st.slider('Mean salary greater than', 30000, 140000, 30000)
    housing_to_filter = st.slider('Mean studio apartment price less than', 451, 2873, 2870)

    filtered_data = grouped_df[(grouped_df['prevailing_wage'] >= wage_to_filter) & (grouped_df['studio_appt_price'] \
        <= housing_to_filter)]
    filtered_bigdata = big_df[(big_df['prevailing_wage'] >= wage_to_filter) & (big_df['studio_appt_price'] <= housing_to_filter)] \
    .sort_values(by='prevailing_wage',ascending=False)

    fig = px.scatter_mapbox(filtered_data, lat="latitude", lon="longitude", hover_name="address", \
        hover_data=["prevailing_wage","studio_appt_price"],
                            color="prevailing_wage", zoom=2.8, height=500, width=100)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    st.plotly_chart(fig)

    if st.checkbox('Show filtered records'):
        cols = ["employer_name","prevailing_wage","employer_state"]
        st_ms = st.multiselect("Columns", big_df.columns.tolist(), default=cols)
        st.write(filtered_bigdata[st_ms])

    st.header('Closer Look')

    score = pd.read_csv('scores.csv')
    option = st.selectbox('What city would you like to explore?',sorted(score['address'].to_list()))

    if st.checkbox('Generate New-Grad Report'):
        plt.figure(figsize=(13,10))
        sns.distplot(big_df[big_df['address']==option]['prevailing_wage'], rug=True,color='red')
        x=big_df[big_df['address']==option]['prevailing_wage'].mean()
        plt.axvline(x=x,ls='--',color='red')
        plt.xlabel('SALARY')
        plt.yticks([])
        st.pyplot()

        filtered_bigdata2 = filtered_bigdata[filtered_bigdata['address']==option]
        if st.checkbox('Show entries'):
            cols = ["employer_name","prevailing_wage","employer_state"]
            st_ms = st.multiselect("Columns", big_df.columns.tolist(), default=cols)
            st.write(filtered_bigdata2[st_ms])

        st.header('Summary of Results')

        arr = filtered_bigdata2['prevailing_wage']
        arr2 = filtered_bigdata2['studio_appt_price']

        temp2 = pd.DataFrame([[option,round(np.mean(arr),2),np.quantile(arr2,0.50),round(np.mean(arr),2)-12*np.quantile(arr2,0.50)\
            ]],columns=['Location','Mean Salary','Median Cost of Studio Apt.','Mean Savings'])
        st.write(temp2)

        st.write('A reasonable salary range for {} is between ${} and ${}, with the mean estimate being ${}.'\
            .format(option,np.min(arr),np.max(arr),round(np.mean(arr),2)))

        st.write('The median cost of a studio appartment in {} is ${}, which equates to an annual cost of ${}.'\
            .format(option, np.quantile(arr2,0.50),12*np.quantile(arr2,0.50)))

        st.header('Evaluation of Job Opportunity')

        st.write('Of the 85 cities we consider in our evaluation, we obtain the following rankings')

        score_temp = score[score['address']==option]
        wage_rank = score_temp['wage_rank']

        temp = pd.DataFrame([[option,score_temp['wage_rank'].tolist()[0], score_temp['appt_rank'].tolist()[0],score_temp['savings_rank'].tolist()[0]\
            ]],columns=['Location','Salary Rank','Cost of Living Rank','Savings Rank'])
        st.write(temp)
        final_score = int(round(score_temp['final_rank'].tolist()[0]))
        st.header('Final Rank: {}'.format(final_score))


        st.write('We observe the final ranking to be {}, which puts it at the {}th percentile of cities. The calculation of final rank is done by calculating the rank product, which is simply the geometric mean of the salary, cost of living and savings ranks. The formula is shown below.'\
            .format(final_score, 
            round(100-(100*((final_score-0.5)/85.0)))
            ))

        st.latex(r'''
        RankProduct(city) = \left(\prod_{i=1}^k r_{city,i}\right)^{\frac{1}{k}}
        ''')


    st.header("New-Grad's Top 10 Picks")

    temp3 = score[['final_rank','address','prevailing_wage','studio_appt_price','savings']].sort_values(by='final_rank').head(10)
    temp3.columns = ['Final Rank', 'City', 'Mean Salary', 'Mean Studio Apt. Cost','Mean Savings']
    temp3.set_index('Final Rank',inplace=True)
    st.write(temp3)

if __name__ == '__main__':
    main()
