const RegisterView = {
    template: `
    <div class="row justify-content-center mt-5">
        <div class="col-md-6">
            <div class="glass-panel">
                <h2 class="text-center mb-4 brand-title">Create Account</h2>
                <div v-if="error" class="alert alert-danger">{{ error }}</div>
                <div v-if="success" class="alert alert-success">{{ success }}</div>
                <form @submit.prevent="register">
                    <div class="mb-3">
                        <label class="form-label">I am a...</label>
                        <select class="form-select" v-model="role">
                            <option value="student">Student</option>
                            <option value="company">Company</option>
                        </select>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">{{ role === 'company' ? 'Company Name' : 'Full Name' }}</label>
                        <input type="text" class="form-control" v-model="name" required>
                    </div>
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
                            <i class="bi bi-person-plus me-2"></i>Register
                        </button>
                    </div>
                </form>
                <div class="text-center mt-3">
                    <small class="text-muted">Already have an account? <router-link to="/login" class="text-info text-decoration-none">Login</router-link></small>
                </div>
            </div>
        </div>
    </div>
    `,
    data() {
        return {
            role: 'student',
            name: '',
            email: '',
            password: '',
            error: null,
            success: null
        }
    },
    methods: {
        async register() {
            try {
                this.error = null;
                this.success = null;
                await axios.post('/api/auth/register', {
                    role: this.role,
                    name: this.name,
                    email: this.email,
                    password: this.password
                });
                this.success = 'Registration successful! You can now login.';
                this.name = ''; this.email = ''; this.password = '';
                setTimeout(() => this.$router.push('/login'), 2000);
            } catch (err) {
                this.error = err.response?.data?.error || 'Registration failed';
            }
        }
    }
}
