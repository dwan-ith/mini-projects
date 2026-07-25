const LoginView = {
    template: `
    <div class="row justify-content-center mt-5">
        <div class="col-md-5">
            <div class="glass-panel">
                <h2 class="text-center mb-4 brand-title">Welcome Back</h2>
                <div v-if="error" class="alert alert-danger">{{ error }}</div>
                <form @submit.prevent="login">
                    <div class="mb-3">
                        <label class="form-label">Email address</label>
                        <input type="email" class="form-control" v-model="email" required>
                    </div>
                    <div class="mb-4">
                        <label class="form-label">Password</label>
                        <input type="password" class="form-control" v-model="password" required>
                    </div>
                    <div class="d-grid">
                        <button type="submit" class="btn btn-primary-gradient py-2">
                            <i class="bi bi-box-arrow-in-right me-2"></i>Login
                        </button>
                    </div>
                </form>
                <div class="text-center mt-4">
                    <small class="text-muted">Don't have an account? <router-link to="/register" class="text-info text-decoration-none">Sign up</router-link></small>
                </div>
            </div>
        </div>
    </div>
    `,
    data() {
        return { email: '', password: '', error: null }
    },
    methods: {
        async login() {
            try {
                this.error = null;
                const res = await axios.post('/api/auth/login', {
                    email: this.email,
                    password: this.password
                });
                localStorage.setItem('token', res.data.token);
                localStorage.setItem('user', JSON.stringify(res.data.user));
                window.dispatchEvent(new Event('login-success'));

                const role = res.data.user.role;
                if (role === 'admin') this.$router.push('/admin');
                else if (role === 'company') this.$router.push('/company');
                else this.$router.push('/student');
            } catch (err) {
                this.error = err.response?.data?.error || 'Login failed';
            }
        }
    }
}
