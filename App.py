import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from io import BytesIO
from fpdf import FPDF
import numpy as np
from collections import Counter, defaultdict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Configuration de la page
st.set_page_config(
    page_title="🧠 Nordique Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .consensus-item {
        background: #d4edda;
        padding: 1rem;
        border-left: 4px solid #28a745;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .discordance-item {
        background: #f8d7da;
        padding: 1rem;
        border-left: 4px solid #dc3545;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        border-radius: 10px;
        font-weight: bold;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

def extract_pdf_text(file):
    """Extrait le texte d'un fichier PDF"""
    try:
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(BytesIO(file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"❌ Erreur lors de la lecture du PDF: {str(e)}")
        return ""

def extract_sentences(text):
    """Extrait les phrases d'un texte"""
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    return sentences

def extract_key_words(documents):
    """Extrait les mots-clés importants de tous les documents"""
    all_text = " ".join(documents)
    vectorizer = TfidfVectorizer(max_features=30, stop_words='english', min_df=1)
    try:
        tfidf_matrix = vectorizer.fit_transform([all_text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        word_scores = list(zip(feature_names, scores))
        word_scores.sort(key=lambda x: x[1], reverse=True)
        return word_scores[:20]
    except:
        return []

def calculate_similarity_matrix(documents):
    """Calcule la matrice de similarité entre documents"""
    if len(documents) < 2:
        return None
    
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        return similarity_matrix
    except:
        return None

def analyze_documents(documents):
    """Analyse les consensus et discordances entre documents"""
    
    # Extraire toutes les phrases de tous les documents
    all_sentences_by_doc = []
    for doc in documents:
        sentences = extract_sentences(doc)
        all_sentences_by_doc.append(sentences)
    
    # Aplatir toutes les phrases
    all_sentences = []
    sentence_to_doc = []
    for doc_idx, sentences in enumerate(all_sentences_by_doc):
        for sentence in sentences:
            all_sentences.append(sentence)
            sentence_to_doc.append(doc_idx)
    
    if len(all_sentences) < 2:
        return None
    
    # Calculer les similarités entre phrases
    vectorizer = TfidfVectorizer(max_features=50, stop_words='english', min_df=1)
    try:
        tfidf_matrix = vectorizer.fit_transform(all_sentences)
    except:
        return {
            "consensus": [],
            "discordances": [],
            "statistics": {
                "total_docs": len(documents),
                "total_sentences": 0,
                "consensus_rate": 0,
                "avg_similarity": 0
            },
            "similarity_matrix": None,
            "key_words": []
        }
    
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Identifier les phrases consensuelles
    consensus_phrases = []
    discordance_phrases = []
    analyzed_phrases = set()
    
    for i, sentence in enumerate(all_sentences):
        if sentence in analyzed_phrases:
            continue
        
        doc_i = sentence_to_doc[i]
        similar_docs = set()
        similarity_scores = []
        
        for j, other_sentence in enumerate(all_sentences):
            if i != j:
                doc_j = sentence_to_doc[j]
                sim_score = similarity_matrix[i][j]
                
                if sim_score > 0.3 and doc_i != doc_j:
                    similar_docs.add(doc_j)
                    similarity_scores.append(sim_score)
        
        if len(similar_docs) >= max(1, len(documents) // 2):
            consensus_phrases.append({
                "phrase": sentence,
                "support_docs": len(similar_docs) + 1,
                "avg_similarity": np.mean(similarity_scores) if similarity_scores else 0,
                "source_doc": doc_i,
                "supporting_docs": list(similar_docs)
            })
            analyzed_phrases.add(sentence)
    
    # Identifier les discordances
    for doc_idx, sentences in enumerate(all_sentences_by_doc):
        for sentence in sentences[:5]:
            if sentence not in analyzed_phrases and len(sentence) > 30:
                discordance_phrases.append({
                    "phrase": sentence,
                    "source_doc": doc_idx,
                    "uniqueness": 1.0
                })
    
    # Trier par pertinence
    consensus_phrases = sorted(consensus_phrases, key=lambda x: x["avg_similarity"], reverse=True)[:15]
    discordance_phrases = sorted(discordance_phrases, key=lambda x: x["uniqueness"], reverse=True)[:15]
    
    # Calculer les statistiques
    doc_similarity = calculate_similarity_matrix(documents)
    avg_similarity = np.mean(doc_similarity) if doc_similarity is not None else 0
    
    # Extraire les mots-clés
    key_words = extract_key_words(documents)
    
    report = {
        "consensus": consensus_phrases,
        "discordances": discordance_phrases,
        "statistics": {
            "total_docs": len(documents),
            "total_sentences": len(all_sentences),
            "consensus_rate": len(consensus_phrases) / max(1, len(consensus_phrases) + len(discordance_phrases)),
            "avg_similarity": float(avg_similarity)
        },
        "similarity_matrix": doc_similarity,
        "key_words": key_words
    }
    
    return report

def plot_consensus_discordance(report):
    """Crée un graphique des consensus vs discordances"""
    consensus_count = len(report["consensus"])
    discordance_count = len(report["discordances"])
    
    fig = go.Figure(data=[
        go.Bar(
            x=['Consensus', 'Discordances'],
            y=[consensus_count, discordance_count],
            marker_color=['#28a745', '#dc3545'],
            text=[consensus_count, discordance_count],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Vue d'ensemble: Consensus vs Discordances",
        xaxis_title="Type",
        yaxis_title="Nombre d'éléments",
        height=400,
        showlegend=False
    )
    
    return fig

def plot_similarity_heatmap(similarity_matrix, num_docs):
    """Crée une heatmap de similarité entre documents"""
    if similarity_matrix is None:
        return None
    
    labels = [f"Doc {i+1}" for i in range(num_docs)]
    
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=labels,
        y=labels,
        colorscale='RdYlGn',
        text=np.round(similarity_matrix, 2),
        texttemplate='%{text}',
        textfont={"size": 12},
        colorbar=dict(title="Similarité")
    ))
    
    fig.update_layout(
        title="🔥 Matrice de Similarité entre Documents",
        xaxis_title="Documents",
        yaxis_title="Documents",
        height=450
    )
    
    return fig

def plot_word_cloud(key_words):
    """Génère un nuage de mots"""
    if not key_words:
        return None
    
    word_freq = {word: score for word, score in key_words}
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',
        relative_scaling=0.5,
        min_font_size=10
    ).generate_from_frequencies(word_freq)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    return fig

def plot_support_distribution(report):
    """Graphique de distribution du support des consensus"""
    if not report["consensus"]:
        return None
    
    support_counts = [item["support_docs"] for item in report["consensus"][:10]]
    phrases = [item["phrase"][:40] + "..." for item in report["consensus"][:10]]
    
    fig = go.Figure(data=[
        go.Bar(
            y=phrases,
            x=support_counts,
            orientation='h',
            marker_color='#667eea',
            text=support_counts,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="📊 Top 10 Consensus - Support par Documents",
        xaxis_title="Nombre de documents supportant",
        yaxis_title="",
        height=500,
        showlegend=False
    )
    
    return fig

def generate_pdf_report(report):
    """Génère un rapport PDF détaillé"""
    pdf = FPDF()
    pdf.add_page()
    
    # Titre
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 15, "Rapport d'Analyse - Consensus/Discordance", ln=True, align='C')
    pdf.ln(5)
    
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 8, "Nordique Analyzer - Analyse Intelligente de Documents", ln=True, align='C')
    pdf.ln(10)
    
    # Statistiques globales
    pdf.set_font("Arial", 'B', 14)
    pdf.set_fill_color(102, 126, 234)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, "Statistiques Globales", ln=True, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)
    
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 8, f"Nombre de documents analyses: {report['statistics']['total_docs']}", ln=True)
    pdf.cell(0, 8, f"Nombre total de phrases: {report['statistics']['total_sentences']}", ln=True)
    pdf.cell(0, 8, f"Taux de consensus: {report['statistics']['consensus_rate']*100:.1f}%", ln=True)
    pdf.cell(0, 8, f"Similarite moyenne: {report['statistics']['avg_similarity']*100:.1f}%", ln=True)
    pdf.ln(10)
    
    # Points de Consensus
    pdf.set_font("Arial", 'B', 14)
    pdf.set_fill_color(40, 167, 69)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, "Points de Consensus", ln=True, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)
    
    pdf.set_font("Arial", '', 10)
    for idx, item in enumerate(report["consensus"][:8], 1):
        phrase = item["phrase"][:100].encode('latin-1', 'replace').decode('latin-1')
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(0, 7, f"{idx}. Support: {item['support_docs']} documents", ln=True)
        pdf.set_font("Arial", '', 9)
        pdf.multi_cell(0, 5, f"   {phrase}")
        pdf.ln(2)
    
    pdf.ln(5)
    
    # Points de Discordance
    pdf.set_font("Arial", 'B', 14)
    pdf.set_fill_color(220, 53, 69)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, "Points de Discordance", ln=True, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)
    
    pdf.set_font("Arial", '', 10)
    for idx, item in enumerate(report["discordances"][:8], 1):
        phrase = item["phrase"][:100].encode('latin-1', 'replace').decode('latin-1')
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(0, 7, f"{idx}. Document source: {item['source_doc'] + 1}", ln=True)
        pdf.set_font("Arial", '', 9)
        pdf.multi_cell(0, 5, f"   {phrase}")
        pdf.ln(2)
    
    # Pied de page
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 5, "Genere par Nordique Analyzer - www.nordique-analyzer.com", ln=True, align='C')
    
    pdf_output = bytes(pdf.output())
    return pdf_output

def load_example_docs():
    """Charge des documents d'exemple sur le changement climatique"""
    return [
        """Le réchauffement climatique est une réalité scientifique indéniable soutenue par 97% des climatologues.
        Les températures moyennes mondiales ont augmenté de plus de 1.2°C depuis l'ère préindustrielle.
        Les énergies renouvelables comme le solaire et l'éolien sont essentielles pour réduire les émissions de CO2.
        L'action climatique doit être une priorité absolue pour tous les gouvernements du monde.
        Les glaciers arctiques fondent à un rythme alarmant, causant une élévation du niveau des mers.
        La transition énergétique nécessite des investissements massifs mais créera des millions d'emplois verts.""",
        
        """Le changement climatique représente le défi le plus urgent de notre époque selon l'ONU.
        Les émissions de gaz à effet de serre doivent être réduites de 50% d'ici 2030 pour limiter le réchauffement.
        Les énergies renouvelables deviennent de plus en plus compétitives économiquement.
        La collaboration internationale est cruciale pour lutter efficacement contre le changement climatique.
        Les phénomènes météorologiques extrêmes augmentent en fréquence et en intensité à cause du réchauffement.
        L'innovation technologique et les solutions vertes sont la clé de notre avenir durable.""",
        
        """Certains experts contestent l'urgence et l'ampleur du réchauffement climatique anthropique.
        Les coûts de la transition énergétique sont prohibitifs et menacent la croissance économique mondiale.
        Les énergies fossiles restent nécessaires pour maintenir notre niveau de vie actuel.
        Les modèles climatiques comportent de nombreuses incertitudes et peuvent être contradictoires.
        L'impact économique des politiques climatiques strictes pourrait dépasser celui du changement lui-même.
        Les cycles naturels du climat expliquent en partie les variations de température observées.""",
        
        """L'innovation technologique peut résoudre la crise climatique sans sacrifier la croissance.
        Les énergies renouvelables sont maintenant moins chères que le charbon dans la plupart des pays.
        La transition écologique représente une opportunité économique majeure pour les entreprises.
        Les jeunes générations exigent une action climatique ambitieuse de la part des dirigeants.
        L'électrification des transports et l'efficacité énergétique sont des solutions immédiates.
        Chaque dixième de degré compte: limiter le réchauffement à 1.5°C plutôt que 2°C éviterait des catastrophes."""
    ]

def main():
    # En-tête principal avec style
    st.markdown('<h1 class="main-header">🧠 Nordique Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyse Intelligente de Consensus et Discordances</p>', unsafe_allow_html=True)
    
    # Sidebar avec informations et paramètres
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/brain.png", width=80)
        st.title("⚙️ Configuration")
        
        st.markdown("---")
        st.markdown("### 📖 À propos")
        st.info("""
        **Nordique Analyzer** utilise l'intelligence artificielle pour identifier:
        - ✅ Les points de consensus entre documents
        - ⚠️ Les points de discordance et désaccords
        - 📊 Les tendances et patterns communs
        """)
        
        st.markdown("---")
        st.markdown("### 🎯 Comment ça marche ?")
        st.markdown("""
        1. **Uploadez** vos documents (TXT ou PDF)
        2. **Analysez** avec notre IA
        3. **Explorez** les résultats interactifs
        4. **Téléchargez** le rapport PDF
        """)
        
        st.markdown("---")
        show_wordcloud = st.checkbox("🌥️ Afficher le nuage de mots", value=True)
        show_support = st.checkbox("📊 Afficher la distribution du support", value=True)
        
        st.markdown("---")
        st.markdown("### 📊 Statistiques")
        st.markdown("*Les statistiques apparaîtront après l'analyse*")
    
    # Zone principale
    st.markdown("---")
    
    # Instructions dans un expander
    with st.expander("ℹ️ Instructions d'utilisation", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### 📁 Étape 1")
            st.write("Uploadez vos documents TXT ou PDF")
        with col2:
            st.markdown("#### 🔍 Étape 2")
            st.write("Cliquez sur 'Analyser' ou essayez l'exemple")
        with col3:
            st.markdown("#### 📊 Étape 3")
            st.write("Explorez les résultats et téléchargez le rapport")
    
    st.markdown("---")
    
    # Zone d'upload et boutons
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "📁 Uploadez vos documents",
            type=["txt", "pdf"],
            accept_multiple_files=True,
            help="Formats supportés: TXT, PDF. Vous pouvez uploader plusieurs fichiers."
        )
    
    with col2:
        st.markdown("##### 🎯 Ou essayez")
        use_example = st.button("📚 Exemple", use_container_width=True)
    
    # Bouton d'analyse principal
    analyze_button = st.button("🔍 Lancer l'Analyse", type="primary", use_container_width=True)
    
    # Logique d'analyse
    documents = []
    doc_names = []
    
    if use_example:
        with st.spinner("📖 Chargement de l'exemple..."):
            documents = load_example_docs()
            doc_names = [f"Exemple {i+1}" for i in range(len(documents))]
            st.success(f"✅ {len(documents)} documents d'exemple chargés!")
    
    elif analyze_button and uploaded_files:
        with st.spinner("📖 Lecture des documents..."):
            for uploaded_file in uploaded_files:
                if uploaded_file.type == "application/pdf":
                    text = extract_pdf_text(uploaded_file)
                else:
                    text = uploaded_file.read().decode("utf-8", errors='ignore')
                
                if text and len(text) > 50:
                    documents.append(text)
                    doc_names.append(uploaded_file.name)
        
        if documents:
            st.success(f"✅ {len(documents)} documents chargés avec succès!")
        else:
            st.error("❌ Aucun document valide n'a pu être chargé.")
    
    elif analyze_button and not uploaded_files:
        st.warning("⚠️ Veuillez d'abord uploader des fichiers ou essayer l'exemple!")
    
    # Analyser et afficher les résultats
    if documents:
        with st.spinner("🔬 Analyse en cours... Cela peut prendre quelques secondes..."):
            report = analyze_documents(documents)
        
        if report is None:
            st.error("❌ Erreur lors de l'analyse. Vérifiez que vos documents contiennent suffisamment de texte.")
            return
        
        # Ligne séparatrice
        st.markdown("---")
        st.markdown("## 📊 Résultats de l'Analyse")
        st.markdown("")
        
        # Métriques en cartes colorées
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📚</h3>
                <h2>{report['statistics']['total_docs']}</h2>
                <p>Documents</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>✅</h3>
                <h2>{len(report['consensus'])}</h2>
                <p>Consensus</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>⚠️</h3>
                <h2>{len(report['discordances'])}</h2>
                <p>Discordances</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📈</h3>
                <h2>{report['statistics']['avg_similarity']*100:.0f}%</h2>
                <p>Similarité</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("")
        st.markdown("---")
        
        # Graphiques de visualisation
        st.markdown("## 📈 Visualisations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_overview = plot_consensus_discordance(report)
            st.plotly_chart(fig_overview, use_container_width=True)
        
        with col2:
            fig_heatmap = plot_similarity_heatmap(
                report['similarity_matrix'],
                report['statistics']['total_docs']
            )
            if fig_heatmap:
                st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Support distribution (optionnel)
        if show_support and report['consensus']:
            st.markdown("---")
            fig_support = plot_support_distribution(report)
            if fig_support:
                st.plotly_chart(fig_support, use_container_width=True)
        
        # Nuage de mots (optionnel)
        if show_wordcloud and report['key_words']:
            st.markdown("---")
            st.markdown("### ☁️ Nuage de Mots-Clés")
            fig_wordcloud = plot_word_cloud(report['key_words'])
            if fig_wordcloud:
                st.pyplot(fig_wordcloud)
        
        st.markdown("---")
        
        # Résultats détaillés en tabs
        st.markdown("## 📋 Résultats Détaillés")
        
        tab1, tab2, tab3 = st.tabs(["✅ Consensus", "⚠️ Discordances", "📊 Mots-Clés"])
        
        with tab1:
            st.markdown("### Points de Consensus Identifiés")
            st.caption("Ces phrases apparaissent de manière similaire dans plusieurs documents")
            
            if report["consensus"]:
                for idx, item in enumerate(report["consensus"], 1):
                    st.markdown(f"""
                    <div class="consensus-item">
                        <strong>#{idx}</strong> - <em>{item['phrase']}</em><br>
                        <small>📊 Support: <strong>{item['support_docs']}</strong> documents | 
                        Similarité: <strong>{item['avg_similarity']:.1%}</strong></small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Aucun consensus significatif détecté entre les documents.")
        
        with tab2:
            st.markdown("### Points de Discordance Identifiés")
            st.caption("Ces phrases sont uniques à certains documents")
            
            if report["discordances"]:
                for idx, item in enumerate(report["discordances"], 1):
                    st.markdown(f"""
                    <div class="discordance-item">
                        <strong>#{idx}</strong> - <em>{item['phrase']}</em><br>
                        <small>📄 Document source: <strong>{doc_names[item['source_doc']] if item['source_doc'] < len(doc_names) else f"Document {item['source_doc'] + 1}"}</strong></small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Aucune discordance majeure détectée entre les documents.")
        
        with tab3:
            st.markdown("### Mots-Clés Principaux")
            st.caption("Les termes les plus importants identifiés dans l'ensemble des documents")
            
            if report["key_words"]:
                # Afficher les mots-clés dans un tableau
                keywords_df = pd.DataFrame(
                    report["key_words"][:15],
                    columns=["Mot-Clé", "Score TF-IDF"]
                )
                keywords_df["Score TF-IDF"] = keywords_df["Score TF-IDF"].apply(lambda x: f"{x:.4f}")
                st.dataframe(keywords_df, use_container_width=True, hide_index=True)
            else:
                st.info("Aucun mot-clé significatif extrait.")
        
        # Bouton de téléchargement du rapport PDF
        st.markdown("---")
        st.markdown("## 📥 Télécharger le Rapport")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            pdf_output = generate_pdf_report(report)
            
            st.download_button(
                label="📄 Télécharger le Rapport PDF Complet",
                data=pdf_output,
                file_name="nordique_analyzer_rapport.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 2rem;'>
        <p><strong>🧠 Nordique Analyzer v2.0</strong></p>
        <p>Analyse Intelligente de Documents par IA</p>
        <p><small>Propulsé par TF-IDF, Scikit-learn & Streamlit</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
