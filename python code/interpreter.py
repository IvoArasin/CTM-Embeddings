# Interpreter - Interactive Browser
import numpy as np#
import pandas as pd#
from sklearn.linear_model import Lasso#
import matplotlib.pyplot as plt#
import networkx as nx #
import dash#
import dash_core_components as dcc#
import dash_html_components as html#
import plotly.graph_objs as go#
from dash.dependencies import Input, Output, State
import random
from dash.exceptions import PreventUpdate
import time

K = 25


def graph(x, size, alpha):
    r = list(np.zeros(size))
    for i in range(size):
        r[i] = int(i)

    x = np.matrix(x)
    Shat = np.full((size, size), 0)
    LAS = Lasso(alpha=alpha, tol=0.0001, fit_intercept=True)
    for i in range(size):
        r2 = r.copy()
        r2.remove(i)
        y = x[:, i]
        X = x[:, r2]
        out = LAS.fit(X, y)
        coefs = out.coef_
        gap = 0
        for j in range(size):
            if j == i:
                Shat[i, i] = 1
                gap += 1
            else:
                if np.abs(coefs[j-gap]) > 0:
                    Shat[i, j] = 1
    G = nx.from_numpy_matrix(np.array(Shat))
    return G

def draw_graph(G, color_map, size_map):
    nx.draw(G, node_color=color_map, node_size=size_map, with_labels=True)
    plt.show()


def clean_string(s):
    s = s.translate(str.maketrans('', '', '[]{}\'\"@.,:;?!=-+/\\&`*#^'))
    return s


def scale(x):
    for i in range(len(x)):
        x[i] = (x[i]-np.mean(x[i]))/(np.full( (1,len(x[i])), np.var(x[i])**0.5)[0])
    return x


def read(path):
    data = np.matrix(pd.read_csv(path, header=None))
    return data


def make_proper(data, length):
    c = np.zeros((len(data), length))
    for i in range(len(data)):
        a = data[i, 0].split()
        b = [float(n) for n in a]
        c[i] = b
    return c


def interpret():
    # scaled Lambda
    # Theta
    # author embeddings
    # Betas as topic-word distributions
    # word_inv list

    word_inv_raw = str(pd.read_csv('words_inv.txt', delimiter='\t'))
    word_inv_raw = word_inv_raw.split()
    words_inv = []
    ind = 3
    for i in range(int(len(word_inv_raw) / 2 - 2)):
        words_inv.append((int(clean_string(word_inv_raw[ind])), clean_string(word_inv_raw[ind + 1])))
        ind += 2
    meta_raw = np.matrix(pd.read_csv('authors.txt', delimiter='\t', header=None))
    meta = []
    au_dir = {}
    au_dir_i = 0
    for i in meta_raw:
        cat_list = i[0, 0].replace(' {', '%[')
        full = cat_list.rsplit('%', 1)
        cats = {}
        cat_list = full[1].translate(str.maketrans('', '', '[]{}\'\":,'))
        cat_list = cat_list.split()
        ind = 0
        for i in range(int(len(cat_list) / 2)):
            cats[cat_list[ind]] = float(cat_list[ind + 1])
            ind += 2
        name_nr = str(full[0]).rsplit(' ', 1)
        name = name_nr[0]
        au_dir[name.replace(' ','')] = au_dir_i
        nr = int(name_nr[1])
        meta.append((name, nr, cats))
        au_dir_i += 1
    cum = np.zeros(len(meta)+1)
    tot = 0
    for i in range(len(cum)-1):
        tot += int(meta[i][1])
        cum[i+1] = tot
    cum = cum.astype(int)
    lam = read('lam_raw.txt')
    lam = make_proper(lam, K)
    theta = lam.copy()
    lam = np.transpose(lam)
    lam = scale(lam)
    lam = np.transpose(lam)

    for i in range(len(theta)):
        theta[i] = np.exp(theta[i])/sum(np.exp(theta[i]))
    auth_embedding = []
    hom = []
    for j in range(len(cum)-1):
        div = np.full((1, K), (cum[j + 1] - cum[j]))[0]
        au_ma = theta[cum[j]:cum[j + 1]]

        aggregate = []
        homogeneity = []
        length = len(au_ma[:, 0])
        #low = int(round(0.3 * length))
        #up = int(round(0.8 * length))
        for k in range(K):
            #print("var", np.var(au_ma[:, k]))
            so = np.mean((au_ma[:, k]))
            aggregate.append(so)
            homogeneity.append(np.std(au_ma[:, k]))

        #for i in range(len(au_ma[:, 0])):
         #   h = sum(aggregate*np.log(aggregate/au_ma[i])+au_ma[i]*np.log(au_ma[i]/aggregate))

          #  homogeneity += h
        #print("AAA", aggregate)
        #aggregate = sum(theta[cum[j]:cum[j + 1]])
            #if div[0] < 5:

            #    auth_embedding.append(np.full((1, K), 100)[0])
            #else:
        #x = aggregate / div
        div = np.full((1, K), sum(aggregate))[0]
        auth_embedding.append(np.array(aggregate)/div)

        hom.append(homogeneity)
    beta = read('beta_numbers.txt')
    beta = make_proper(beta, len(words_inv))

    topic_words = []
    for j in range(K):
        s = list(beta[j])
        s_order = []
        for i in range(15):
            rounded = round(max(s) * 1000) / 1000
            s_order.append((words_inv[s.index(max(s))][1]))
            s[s.index(max(s))] = -1000
        #s_order = np.concatenate(s_order)
        topic_words.append(s_order)

    return meta, theta, beta, words_inv, lam, auth_embedding, topic_words, au_dir, hom

meta, theta, beta, words_inv, lam, auth_embedding, topic_words, au_dir, hom = interpret()

np.savetxt("auth_embed.txt", auth_embedding)
#xxx = re.sub(r' +', ',', str(sum(theta)))


def nx_to_plotly(G, node_cols):
    pos = nx.spring_layout(G)
    node_x = []
    node_y = []
    text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        text.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y, text=text,
        mode='markers+text',
        showlegend=False,
        hoverinfo='text',
        marker=dict(color=node_cols,
                    size=50,
                    line=dict(color='white', width=0)))

    hover_text = []
    for i in range(len(G.nodes())):
        k_w_html = ''
        for k_word in topic_words[i]:
            k_w_html += str(k_word)+'<br>'
        hover_text.append(k_w_html)

    node_trace['hovertext'] = hover_text

    layout = dict(plot_bgcolor='rgb(225,225,245)', #'rgba(18,60,105, 0.3)',
                  paper_bgcolor='rgba(18,60,105, 0.0)',
                  margin=dict(t=0, b=0,l=0,r=0, pad=0),
                  xaxis=dict(linecolor='black',
                             showgrid=False,
                             showticklabels=False,
                             mirror=False,
                             visible=False),
                  yaxis = dict(linecolor='black',
                    showgrid=False,
                    showticklabels=False,
                    mirror=False,
                    visible=False))

    ############
    fig = go.Figure(layout=layout)

    ex_tuples = []
    ey_tuples = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        ex = []
        ey = []
        ex.append(x0)
        ex.append(x1)
        ex.append(None)
        ey.append(y0)
        ey.append(y1)
        ey.append(None)
        ex_tuples.append(ex)
        ey_tuples.append(ey)
    line_cols = ['rgb(112,128,144)' for i in range(len(ex_tuples))]

    for idx in range(len(ex_tuples)):
        fig.add_trace(go.Scatter(
            x=ex_tuples[idx],
            y=ey_tuples[idx],
            line=dict(
                width=2,
                color=line_cols[idx],
            ),
            hoverinfo='none',
            showlegend=False,
            mode='lines'
        ))

    fig.add_traces(node_trace)
    #print(pos)
    #print(ex_tuples)
    #print(ey_tuples)
    ##############
    #fig = go.Figure(data=[edge_trace, node_trace], layout=layout)

    return fig, node_trace, layout, ex_tuples, ey_tuples, pos


def K_edge_dict(pos, ex_tuples):
    k_edge_dict = {}
    #ind = 0
    for node in G.nodes():
        target = pos[node][0]
        target_l = len(ex_tuples)
        trace_set = []
        for ind in range(target_l):
            if ex_tuples[int(ind)][0] == target or ex_tuples[int(ind)][1] == target:
                trace_set.append(int(ind))
            #ind += 1
        k_edge_dict[node] = trace_set[1:]
    return k_edge_dict


def update_edges(node_trace, layout, ex_lines, ey_lines, line_cols):
    fig = go.Figure(layout=layout)
    if not isinstance(line_cols, str):
        for ind, node in enumerate(line_cols):
            if node >= 0.1:
                for idx in k_edge_dict[ind]:
                    fig.add_trace(go.Scatter(
                        x=ex_lines[idx],
                        y=ey_lines[idx],
                        line=dict(
                            width=2,
                            color='rgba(112,128,144, 0.7)',
                        ),
                        hoverinfo='none',
                        showlegend=False,
                        mode='lines'
                    ))
        fig.add_traces(node_trace)
    else:
        for idx in range(len(ex_lines)):
            fig.add_trace(go.Scatter(
                x=ex_lines[idx],
                y=ey_lines[idx],
                line=dict(
                    width=2,
                    color='rgb(112,128,144)',
                ),
                hoverinfo='none',
                showlegend=False,
                mode='lines'
            ))
        fig.add_traces(node_trace)
    return fig

#(xxx)
#print(np.matrix(auth_embedding))
#auth_embedding_trans = np.transpose(np.matrix(auth_embedding))
#auth_embedding_trans_scaled = scale(auth_embedding_trans)
#print(auth_embedding_trans_scaled)
#graph(auth_embedding_trans_scaled, 97)


def bucket(a,b):

    level = 0
    for i in b:
        if i in a:
            if b[i] > a[i]:
                level += a[i]
            else:
                level += b[i]
    return level


def bucket_broad(a, b):
    a_broad = {}
    for i in a:
        if (i.split('.'))[0] in a_broad:
            a_broad[(i.split('.'))[0]] += a[i]
        else:
            a_broad[(i.split('.'))[0]] = a[i]
    b_broad = {}
    for i in b:
        if (i.split('.'))[0] in b_broad:
            b_broad[(i.split('.'))[0]] += b[i]
        else:
            b_broad[(i.split('.'))[0]] = b[i]
    level = 0
    for i in b_broad:
        if i in a_broad:
            if b_broad[i] > a_broad[i]:
                level += a_broad[i]
            else:
                level += b_broad[i]
    return level


def ndcg(x, base):
    x = x[1:]
    x_ideal = np.sort(x)
    x_ideal = x_ideal[::-1]
    if x_ideal[0] != 0:
        dcg_normaliser = 0
        dcg = 0
        for i, j in enumerate(x_ideal):
            dcg_normaliser += j/(np.log(i+2)/np.log(base))
        for i, j in enumerate(x):
            dcg += j/(np.log(i+2)/np.log(base))

        return dcg/dcg_normaliser
    else:
        return 0


last_dist = []
def dist(a,b, min_filling, inv, bucket_size):
    all_lvl = []
    distTo = b[a]
    a1 = meta[a][2]
    neighbours = []
    dist = []
    m = meta.copy()
    hom_i = 0
    for o in b:  # Hellinger Distance for discrete probability distributions
        d = (1 / (2 ** 0.5)) * (sum(  ((distTo ** 0.5 - o ** 0.5) ** 2))) ** 0.5
        #d = sum(distTo*np.log(distTo/o)+o*np.log(o/distTo))
        # d = round(d*1000)/1000
        dist.append(d)
        hom_i += 1
    mi_old = -1
    #lvl_list = np.zeros(len(meta)-1)
    lvl_list = []
    if inv == True:
        mima = np.max
    else:
        mima = np.min
    if bucket_size == 'broad':
        buckets = bucket_broad
    elif bucket_size == 'narrow':
        buckets = bucket
    else:
        buckets = 'none'
    #print(dist)
    for i in range(len(meta)-1):
        mi = mima(dist)
        ind = dist.index(mi)
        if not buckets == 'none':
            lvl = buckets(a1, m[ind][2])
            lvl = round(1000*lvl)/1000
            #neighbours =((m[ind][0], mi, lvl, m[ind][2]))
            #lvl_list[i] = float(lvl)
            lvl_list.append(float(lvl))
            if lvl < min_filling:
                if len(neighbours)-1 > 0:
                    #print(len(neighbours)-1, ' author(s) found within proximity ', min_filling)
                    last_dist.append(mi_old)
                    all_lvl.append(np.mean(lvl_list))

                break

        neighbours.append((m[ind][0], (np.round(1000*mi)/1000))) #, lvl, m[ind][2]))
        mi_old = mi
        dist.remove(mi)
        del m[ind]
    all_lvl.append(lvl_list) # ndcg(lvl_list, 2)))
    ndcg_ = -1
    if not buckets == 'none':
        ndcg_ = ndcg(all_lvl[0], 2)
    #print('\n')
    #print(neighbours)
    #print(all_lvl)
    #print(lvl_list)

    return lvl_list, neighbours, ndcg_


ndcg_tot = 0
total = [] # [32, 66, 1, 31, 143, 123, 10, 34, 193, 222]
for i in range(len(auth_embedding)):
    rand = np.floor(238*random.uniform(0,1))
    a, b, c = dist(int(rand), auth_embedding, 0, False, 'broad')
    #print(sum(auth_embedding[i]))
    total.append(np.array(a))
    #print(rand, b[0:2], b[int(100+np.floor(100*random.uniform(0,1)))])
    ndcg_tot += c
ndcg_tot = ndcg_tot/len(auth_embedding)
#print(ndcg_tot)
#print(re.sub(r' +', ',', str(sum(total))))

e = """
r_nr = []
for i in ['Beom Jun Kim','A. V. Tsiganov','Alex Townsend','Laurent Fargues','I. S. Timchenko','K. Nimmo','Michel Habib','Christophe De Wagter','Jennifer Cano','Amr S. Helmy']:
    rand = i #np.floor(random.uniform(0, 1)*238)
    r_nr.append(int(rand))
    a, b, c = dist(int(rand), auth_embedding, 0, False, 'narrow')
    print(rand, b[0:2])# , b[int(100+np.floor(random.uniform(0, 1)*100))])
print(r_nr)

recs = []
for i in ['Beom Jun Kim','A. V. Tsiganov','Alex Townsend','Laurent Fargues','I. S. Timchenko','K. Nimmo','Michel Habib','Christophe De Wagter','Jennifer Cano','Amr S. Helmy']:
    ind = [t[0] for t in meta].index(i)
    print(ind)
    a, b, c = dist(ind, auth_embedding, 0, False, 'narrow')
    total.append(np.array(a))
    ran = int(100+np.floor(100*random.uniform(0,1)))
    ndcg_tot += c
    recs.append((b[0][0], b[1][0], b[2][0], b[ran][0]))
print(recs)
"""

def node_config(input, color):
    heat_map = []
    border_map = []
    r, g, b = 254, 156, 143
    if color == 'green':
        r, g, b = 76, 154, 42
    for k in range(len(input)):
        col = (r, g, b, input[k])
        heat_map.append('rgba'+str(col))
        if color == 'green' and input[k] >= 0.4:
            border_map.append(3)
        elif color == 'green' and input[k] >= 0.2:
            border_map.append(1)
        else:
            border_map.append(0)
        #size_map.append((300*input[k]*2))
    return heat_map, border_map #, size_map


def flip(input):
    words = {}
    for i in range(len(input)):
        words[input[i][1]] = i
    return words


words = flip(words_inv)


def au_recom(str_input):
    if str_input == '':
        heat_map_and_authors = 'rgba(254,156,143, 1)', '', 'no'
    elif str_input.replace(' ','') in au_dir:
        str_input = str_input.replace(' ','')
        author_i = auth_embedding[int(au_dir[str_input])]
        author_i_heat = []
        for au_k in author_i:
            au_k_heat = 0.01 + au_k/np.max(author_i)
            if au_k_heat <= 1:
                author_i_heat.append(au_k_heat)
            else:
                author_i_heat.append(1)
        a, b, c = dist(int(au_dir[str_input]), auth_embedding, 0, False, 'none')
        author_recom_author = []
        for bs in b: # .translate(str.maketrans('', '', '.'))    .replace(' ', '+')
            str_number = str(bs[1])
            while len(str_number) < 5:
                str_number = str_number+'0'
            author_recom_author.append(html.A(str_number+' '+str(bs[0]),
                                href='https://arxiv.org/search/?query=%22' +
                                      ' '+bs[0]+'%22&searchtype=author&source=header',
                                 style={'font-family':'Arial, sans-serif', 'font-weight': 'bold', 'color': 'black','text-decoration': 'none'}, target='_blank'))
            author_recom_author.append(html.Br())
        heat_map_and_authors = author_i_heat, author_recom_author, 'green'
    else:
        query = clean_string(str_input)
        query = query.lower()
        query = query.split()
        query_lkhood = []
        author_lk = []
        author_words = []
        for k in range(K):

            lk = 0
            for q in query:
                try:
                    lk += beta[k, words[q.lower()]]
                except:
                    lk += 0
            if lk == 0:
                heat_map_and_authors = 'rgba(254,156,143, 1)', '', 'no'
                return heat_map_and_authors

            query_lkhood.append(lk)
        query_max = np.max(query_lkhood)

        for a in auth_embedding:
            if not a[0] == 100:
            #author_lk.append(np.dot((np.dot(np.diag(a), query_lkhood)), np.ones(K)))
            # (1/(2**0.5))*(sum((query_lkhood**0.5-a**0.5)**2))**0.5
                author_lk.append((1/(2**0.5))*( sum( (np.array(query_lkhood)**0.5-a**0.5)**2)**0.5 ) )

            else:
                author_lk.append(100)
        author_lk_sorted_names = []
        meta_copy = meta.copy()
        for i in range(len(meta)):
            #ma = np.max(author_lk)
            ma = np.min(author_lk)
            ma_ind = author_lk.index(ma)
            #author_lk_sorted_names.append(meta_copy[ma_ind][0])
            author_lk_sorted_names.append(html.A(' '+str(meta_copy[ma_ind][0]), href='https://arxiv.org/search/?query=%22'+str((meta_copy[ma_ind][0].translate(str.maketrans('', '', '.'))).replace(' ','+'))+'%22&searchtype=author&source=header', style={'font-family':'Arial, sans-serif', 'font-weight':'bold', 'color':'black','text-decoration':'none'}, target='_blank'))
            author_lk_sorted_names.append(html.Br())
            author_lk.remove(ma)
            del meta_copy[ma_ind]
        for q in range(len(query_lkhood)):
            query_lkhood[q] = 0.01+query_lkhood[q]/query_max  # round(10000*query_lkhood[q])/10000
            if query_lkhood[q] > 1:
                query_lkhood[q] = 1
        heat_map_and_authors = query_lkhood, author_lk_sorted_names, 'no'
    return heat_map_and_authors

def k_node_authors(k):
    # most likely authors given one specific topic
    auth_embedding_k = []
    auth_embedding_k_sorted = []
    meta_copy = meta.copy()
    for a in auth_embedding:
        sub = [a[i] for i in k]
        auth_embedding_k.append(sum(sub))
    for ak in range(len(meta)):
        m = np.max(auth_embedding_k)
        m_ind = auth_embedding_k.index(m)
        auth_embedding_k_sorted.append(html.A(' '+str(meta_copy[m_ind][0]), href='https://arxiv.org/search/?query=%22'+str((meta_copy[m_ind][0].translate(str.maketrans('', '', '.'))).replace(' ','+'))+'%22&searchtype=author&source=header', style={'font-family':'Arial, sans-serif', 'font-weight':'bold', 'color':'black','text-decoration':'none'}, target='_blank'))
        auth_embedding_k_sorted.append(html.Br())
        auth_embedding_k.remove(m)
        del meta_copy[m_ind]

    return auth_embedding_k_sorted

# Globals
G = graph(lam, K, 0.2)
G_drawn, node_trace, layout, ex, ey, pos = nx_to_plotly(G, 'rgba(254,156,143, 1)')
k_edge_dict = K_edge_dict(pos, ex)
container = []
container_cols = ['rgba(254,156,143, 0.2)' for i in range(K)]
ret_store = G_drawn, [], html.H3('Recommendations for Query', style={'font-family':'Arial, sans-serif'}), False
topicSearch = False
#post_activation = False
post_activation_clicks = 0


app = dash.Dash(__name__)
app.title = 'Visual Topic Model'

app.layout = html.Div([
    html.Div([dcc.Graph(id='G_network', figure=G_drawn),
              dcc.ConfirmDialog(id='no_match', message='No match found'),
              ]),
    html.Div(
        className="slider_container", style={'font-family':'Arial, sans-serif', 'font-weight': 'bold'},
        children=[
            dcc.Slider(
                id='alpha_slider',
                min=0.05,
                max=1,
                step=0.05,
                value=0.2

            ),
            html.Div(id='alpha_slider_output', style={'margin-left':'1%'})
        ]
    ),
    html.Div([   html.Button('Reset', id='button_search', n_clicks=0, hidden=False, style={'background-color':'rgba(18,60,105, 0.2)'}),
                dcc.Input(id='searchbar', type='text', value='', placeholder='enter query', debounce=True,
                style={'width': '70%'}),
                 html.Button('Topic Search', value="one", id='topical_search', n_clicks=0, hidden=False, style={'background-color':'rgba(18,60,105, 0.2)'})], id="div_searchbar", style={'text-align':'center', 'margin-top':'2%','margin-bottom':'2%', 'margin': 'auto'}),  # , style=dict(display='none'))


    html.Div([
        html.Div([
            html.Div(id='left-recommender-column'),
            html.Div(id='cum_LK_author_for_query')
            ], style={'display': 'inline-block','width':'49%','margin-left':'1%'}),
        html.Div([
            html.Div(id='author-topic-node'),
            html.Div(id='recommendations')
            ], style={'display': 'inline-block','width':'49%','margin-right':'1%'})
      ], style={'width':'100%','margin':'auto'})


], style={'background-color':'rgba(225,225,245, 1)'}) #18,60,105, 0.3

#app.scripts.append_script({
#    'external_url': '/assets/style.css'
#})

@app.callback(
    Output(component_id='G_network', component_property='figure'),
    Output(component_id='cum_LK_author_for_query',component_property='children'),
    Output(component_id='left-recommender-column', component_property='children'),
    Output(component_id='no_match', component_property='displayed'),
    Input(component_id='searchbar', component_property='value'),
    Input(component_id='alpha_slider', component_property='value'),
    Input(component_id='G_network', component_property='clickData'),
    Input(component_id='topical_search', component_property='n_clicks'),
    state=[State('topical_search', 'value')],
    )
def grapher(query, a, clickData, n_clicks, value):
    global topicSearch, G_drawn, container, container_cols, ret_store, post_activation_clicks
    global G_drawn, node_trace, layout, ex, ey, pos, k_edge_dict
    ctx = dash.callback_context
    button_diffferentiatior = ctx.triggered[0]['prop_id'].split('.')[0]
    #print(button_diffferentiatior)
    h3 = html.H3('Recommendations for Query', style={'font-family':'Arial, sans-serif'})
    ret = G_drawn, [], h3, False
    if ctx.triggered[0]['prop_id'] == 'searchbar.value':
        #print("0")
        q_res, author_query_LK, green = au_recom(query)
        col, border = node_config(q_res, green)

        if q_res == 'rgba(254,156,143, 1)' and green=='no' and query is not "":
            #print("2")
            G_drawn.update_traces(marker=dict(color=q_res, line=dict(width=0)))
            ret = G_drawn, [], h3, True

        elif q_res == 'rgba(254,156,143, 1)':
            #print("3")
            #global topicSearch
            topicSearch = False
            post_activation_clicks = 0
            container = []
            container_cols = ['rgba(254,156,143, 0.2)' for i in range(K)]
            G_drawn = update_edges(node_trace, layout, ex, ey, 'rgb(112,128,144)')
            G_drawn.update_traces(marker=dict(color='rgba(254,156,143, 1)', line=dict(width=0)))
            ret_store = G_drawn, [], html.H3('Recommendations for Query', style={'font-family':'Arial, sans-serif'}), False
            ret = G_drawn, [], h3, False

        else:
            #print("4")
            G_drawn = update_edges(node_trace, layout, ex, ey, q_res)
            G_drawn.update_traces(marker=dict(color=col, line=dict(width=border)))
            if green == 'green':
                ret_store = G_drawn, author_query_LK, html.H3('Recommendations for Author'), False
            else:
                ret_store = G_drawn, author_query_LK, h3, False
            ret = ret_store

    elif (button_diffferentiatior == 'G_network' and topicSearch is True):
        #print("5")
        node_k = str(clickData).replace("text': '", '_x_x_')
        node_k = (node_k.split('_x_x_'))[1]
        node_k = (node_k.split("'", 1))[0]
        if int(node_k) in container:
            container.remove(int(node_k))
            container_cols[int(node_k)] = 'rgba(254,156,143, 0.2)'
            G_drawn.update_traces(marker=dict(color=container_cols, line=dict(width=0)))
        else:
            container.append(int(node_k))
            container_cols[int(node_k)] = 'rgba(254,156,143, 1)'
            G_drawn.update_traces(marker=dict(color=container_cols, line=dict(width=0)))
        ret = G_drawn, [], h3, False
    elif button_diffferentiatior == "topical_search":
        topicSearch = True
        #print("6")
        if post_activation_clicks == 0:
            post_activation_clicks = 1
            G_drawn = update_edges(node_trace, layout, ex, ey, 'rgb(112,128,144)')
            G_drawn.update_traces(marker=dict(color="rgba(254,156,143, 0.2)", line=dict(width=0)))
            ret = G_drawn, [], h3, False
    elif button_diffferentiatior=='alpha_slider':
        #print("8")
        G_drawn, node_trace, layout, ex, ey, pos = nx_to_plotly(graph(lam, K, a), 'rgba(254,156,143, 1)')
        k_edge_dict = K_edge_dict(pos, ex)
        ret = G_drawn, [], h3, False
    else:
        #print("7")
        ret = ret_store
    return ret

@app.callback(
    Output(component_id='alpha_slider_output',component_property='children'),
    Input(component_id='alpha_slider',component_property='value')
)
def selected_alpha(a):
    return 'Sparsity: '+str(a)


@app.callback(
    Output('recommendations', 'children'),
    Output('author-topic-node', 'children'),
    Input('G_network', 'clickData'),
    Input(component_id='button_search', component_property='n_clicks')
)
def on_click(clickData, n_clicks):
    ctx = dash.callback_context
    button_diffferentiatior = ctx.triggered[0]['prop_id'].split('.')[0]
    ret = [], html.H3('Most likely authors for topic node none', style={'font-family':'Arial, sans-serif'})
    time.sleep(0.25)
    if button_diffferentiatior == 'G_network':
        node_k = str(clickData).replace("text': '", '_x_x_')
        #print(node_k)
        node_k = (node_k.split('_x_x_'))[1]
        node_k = (node_k.split("'", 1))[0]

        if topicSearch is False:
            res = k_node_authors([int(node_k)])
            h3 = html.H3('Most likely authors for topic node '+str(int(node_k)), style={'font-family':'Arial, sans-serif'})
            ret = res, h3
        elif post_activation_clicks == 1:
            if len(container) > 0:
                res = k_node_authors(container)
            else:
                res = []
            h3 = html.H3('Most likely authors for topic search', style={'font-family':'Arial, sans-serif'})
            ret = res, h3
    elif ctx.triggered[0]['prop_id'] == 'button_search.n_clicks':
        ret = [], html.H3('Most likely authors for topic node none', style={'font-family':'Arial, sans-serif'})


    return ret

@app.callback(
    Output(component_id='searchbar', component_property='value'),
    Input(component_id='button_search', component_property='n_clicks'),
    state=[State('button_search', 'value')]
)
def b(n_clicks, value):
    if n_clicks == 0:
        raise PreventUpdate
    else:
        return ''


if __name__ == '__main__':
   app.run_server(debug=True)



