import pandas as pd

class BatchPredictor:
    def __init__(self, model):
        self.model = model

    def _get_discount(self, prob):
        if prob <= 0.85: return 0.05
        elif prob <= 0.95: return 0.10
        else: return 0.15

    def generate_retention_leads(self, X_new, original_df, output_csv='data/processed/monthly_retention_leads.csv'):
        print("4. Predictor: Finding At-Risk Customers and calculating ROI...")
        
        churn_probs = self.model.predict_proba(X_new)[:, 1]
        
        leads = original_df.copy()
        leads['Churn_Prob'] = churn_probs
        high_risk = leads[leads['Churn_Prob'] > 0.75].copy()
        
        if 'Monthly Charge' in high_risk.columns:
            high_risk = high_risk.rename(columns={'Monthly Charge': 'Monthly_Charge'})
            
        high_risk['Discount_Rate'] = high_risk['Churn_Prob'].apply(self._get_discount)
        high_risk['Monthly_Discount'] = high_risk['Monthly_Charge'] * high_risk['Discount_Rate']
        high_risk['Annual_Offer_Cost'] = high_risk['Monthly_Discount'] * 12
        high_risk['Expected_Loss'] = high_risk['Churn_Prob'] * high_risk['CLTV']
        high_risk['Net_ROI'] = high_risk['Expected_Loss'] - high_risk['Annual_Offer_Cost']
        
        high_risk = high_risk.sort_values('Expected_Loss', ascending=False)
        high_risk['Priority_Rank'] = range(1, len(high_risk) + 1)
        
        output_cols = ['Priority_Rank', 'Churn_Prob']
        if 'Contract' in high_risk.columns:
            output_cols.append('Contract')
        output_cols.extend(['CLTV', 'Monthly_Charge', 'Discount_Rate', 
                            'Monthly_Discount', 'Annual_Offer_Cost', 
                            'Expected_Loss', 'Net_ROI'])
        # Keep only the columns that actually exist in original_df
        output_cols = [col for col in output_cols if col in high_risk.columns]
        
        import os
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        high_risk[output_cols].to_csv(output_csv, index=False)
        print("\n")
        print(f"Customers flagged (>75% risk) : {len(high_risk)}")
        print(f"Total revenue at risk: ${high_risk['Expected_Loss'].sum():,.0f}")
        print(f"Total cost of all offers: ${high_risk['Annual_Offer_Cost'].sum():,.0f}")
        print(f"Net ROI if all retained: ${high_risk['Net_ROI'].sum():,.0f}")
        print("\nTop 5 Priority Customers:")
        print(high_risk[output_cols].head().to_string(index=False))
        
        return high_risk