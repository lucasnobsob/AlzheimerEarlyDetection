import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_outliers(data, column):
    """
    Analisa os outliers de uma coluna e sugere técnicas de tratamento.
    
    Args:
        data: DataFrame
        column: Nome da coluna a ser analisada
    """
    # Cálculo do IQR
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    # Limites para outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identificação de outliers
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)][column]
    
    print(f"\nAnálise de Outliers para {column}:")
    print(f"Total de outliers: {len(outliers)}")
    print(f"Percentual de outliers: {(len(outliers)/len(data))*100:.2f}%")
    print(f"Valores mínimos dos outliers: {outliers.min()}")
    print(f"Valores máximos dos outliers: {outliers.max()}")
    
    # Plot da distribuição
    #plt.figure(figsize=(10, 4))
    #plt.subplot(1, 2, 1)
    #sns.boxplot(y=data[column])
    #plt.title(f'Box Plot - {column}')
    #
    #plt.subplot(1, 2, 2)
    #sns.histplot(data[column], kde=True)
    #plt.title(f'Histograma - {column}')
    #plt.tight_layout()
    #plt.show()
    
    return {
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'outliers': outliers
    }

def winsorize_column(data, column, lower_percentile=5, upper_percentile=95):
    """
    Aplica Winsorization em uma coluna.
    
    Args:
        data: DataFrame
        column: Nome da coluna
        lower_percentile: Percentil inferior
        upper_percentile: Percentil superior
    """
    lower = np.percentile(data[column], lower_percentile)
    upper = np.percentile(data[column], upper_percentile)
    
    data[column] = data[column].clip(lower=lower, upper=upper)
    return data

def apply_log_transform(data, column):
    """
    Aplica transformação logarítmica em uma coluna.
    
    Args:
        data: DataFrame
        column: Nome da coluna
    """
    # Adiciona 1 para evitar log(0)
    data[column] = np.log1p(data[column])
    return data

def impute_by_median(data, column):
    """
    Substitui outliers pela mediana da coluna.
    
    Args:
        data: DataFrame
        column: Nome da coluna
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    median = data[column].median()
    data.loc[data[column] < lower_bound, column] = median
    data.loc[data[column] > upper_bound, column] = median
    
    return data

def remove_multiple_outliers(data, columns, threshold=2):
    """
    Remove linhas que possuem outliers em múltiplas colunas.
    
    Args:
        data: DataFrame
        columns: Lista de colunas para verificar outliers
        threshold: Número mínimo de outliers para remover a linha
    """
    outlier_mask = pd.Series(False, index=data.index)
    
    for column in columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        column_outliers = (data[column] < lower_bound) | (data[column] > upper_bound)
        outlier_mask = outlier_mask | column_outliers
    
    # Conta o número de outliers por linha
    outlier_count = outlier_mask.groupby(outlier_mask.index).sum()
    
    # Remove linhas com mais outliers que o threshold
    rows_to_keep = outlier_count[outlier_count < threshold].index
    return data.loc[rows_to_keep]

def suggest_treatment(data, column):
    """
    Sugere o melhor tratamento para outliers baseado nas características da coluna.
    
    Args:
        data: DataFrame
        column: Nome da coluna
    """
    analysis = analyze_outliers(data, column)
    n_outliers = len(analysis['outliers'])
    total_rows = len(data)
    outlier_percentage = (n_outliers / total_rows) * 100
    
    print("\nSugestões de tratamento:")
    
    if outlier_percentage < 5:
        print("1. Manter os outliers - são poucos e podem ser importantes")
    elif outlier_percentage < 20:
        print("1. Winsorization - moderado número de outliers")
        print("2. Imputação por mediana - preserva a distribuição")
    else:
        print("1. Transformação logarítmica - muitos outliers")
        print("2. Remoção seletiva - se os outliers forem claramente erros")
    
    # Verifica se a distribuição é assimétrica
    skewness = data[column].skew()
    if abs(skewness) > 1:
        print("3. Considerar transformação logarítmica devido à assimetria")
    
    return analysis