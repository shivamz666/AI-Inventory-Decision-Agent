import urllib.request
import urllib.error
import json
import sys

BASE = 'http://127.0.0.1:8000'

def get(path):
    req = urllib.request.Request(f'{BASE}{path}')
    with urllib.request.urlopen(req) as res:
        return res.status, json.loads(res.read().decode('utf-8'))

def post(path, body):
    data = json.dumps(body).encode('utf-8')
    req = urllib.request.Request(f'{BASE}{path}', data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as res:
        return res.status, json.loads(res.read().decode('utf-8'))

def main():
    print("Testing API endpoints on " + BASE + "...")
    # 1. Health
    s, d = get('/health')
    assert s == 200 and d['status'] == 'healthy', f'Health failed: {d}'
    print('[PASS] /health')

    # 2. Products
    s, d = get('/products')
    assert s == 200 and len(d['products']) == 20, f'Products list failed: {d}'
    print(f'[PASS] /products ({len(d["products"])} items)')

    # 3. Product detail
    s, d = get('/products/P019')
    assert s == 200 and d['decision']['decision'] == 'INCREASE', f'P019 failed: {d}'
    print(f'[PASS] /products/P019 (INCREASE, Rec Qty: {d["decision"]["recommended_order_quantity"]})')

    s, d = get('/products/P001')
    assert s == 200 and d['decision']['decision'] == 'REDUCE', f'P001 failed: {d}'
    print(f'[PASS] /products/P001 (REDUCE, Rec Qty: {d["decision"]["recommended_order_quantity"]})')

    # 4. POST /decision
    s, d = post('/decision', {
        'product_id': 'CUSTOM-TEST',
        'forecast_7d': 70,
        'current_inventory': 10,
        'lead_time_days': 7,
        'minimum_order_quantity': 50
    })
    assert s == 200 and d['decision'] == 'INCREASE', f'/decision failed: {d}'
    print('[PASS] POST /decision')

    # 5. POST /analyze
    s, d = post('/analyze', {
        'product_id': 'CUSTOM-ANALYZE',
        'forecast_7d': 70,
        'current_inventory': 500,
        'lead_time_days': 7,
        'minimum_order_quantity': 50
    })
    assert s == 200 and 'ai_explanation' in d, f'/analyze failed: {d}'
    print('[PASS] POST /analyze')

    # 6. POST /alternative-products
    s, d = post('/alternative-products', {
        'product_id': 'P001',
        'category': 'Computer Accessories',
        'forecast_7d': 10,
        'sales_trend': 'decreasing',
        'current_inventory': 5000
    })
    assert s == 200 and isinstance(d, list), f'/alternative-products failed: {d}'
    print(f'[PASS] POST /alternative-products ({len(d)} alternatives found)')

    # 7. POST /approval
    s, d = post('/approval', {
        'product_id': 'P001',
        'decision': 'REDUCE',
        'action': 'APPROVE'
    })
    assert s == 200 and d['action'] == 'APPROVE', f'/approval failed: {d}'
    print('[PASS] POST /approval')

    # 8. GET /approvals
    s, d = get('/approvals')
    assert s == 200 and len(d) >= 1, f'/approvals failed: {d}'
    print(f'[PASS] GET /approvals ({len(d)} recorded)')

    # 9. GET /demo/scenarios
    s, d = get('/demo/scenarios')
    assert s == 200 and len(d) == 3, f'/demo/scenarios failed: {d}'
    print('[PASS] GET /demo/scenarios (3 scenarios)')

    # 10. Error cases
    try:
        get('/products/INVALID_ID_999')
        print('[FAIL] 404 not returned')
        sys.exit(1)
    except urllib.error.HTTPError as e:
        assert e.code == 404, f'Expected 404, got {e.code}'
        print('[PASS] 404 on invalid product_id')

    try:
        post('/approval', {'product_id': 'P001', 'decision': 'REDUCE', 'action': 'INVALID_ACTION'})
        print('[FAIL] 400 not returned')
        sys.exit(1)
    except urllib.error.HTTPError as e:
        assert e.code == 400, f'Expected 400, got {e.code}'
        print('[PASS] 400 on invalid approval action')

    print('\n>>> ALL 10 ENDPOINT SUITES VERIFIED WITH 0 ERRORS! <<<')

if __name__ == '__main__':
    main()
