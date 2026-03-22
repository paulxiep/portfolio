-- Seed data: two test tenants for development and isolation testing

INSERT INTO tenants (id, name, config, rate_limit) VALUES
    ('a0000000-0000-0000-0000-000000000001', 'Test Tenant Alpha', '{"description": "Primary test tenant"}', 60),
    ('b0000000-0000-0000-0000-000000000002', 'Test Tenant Beta', '{"description": "Secondary test tenant for isolation testing"}', 30)
ON CONFLICT (id) DO NOTHING;
