const NavbarComponent = {
    template: `
    <nav class="navbar navbar-expand-lg navbar-dark navbar-glass fixed-top">
        <div class="container">
            <a class="navbar-brand brand-title fw-bold" href="#/">
                <i class="bi bi-rocket-takeoff me-2"></i>PPA V2
            </a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ms-auto" v-if="!user">
                    <li class="nav-item">
                        <a class="nav-link" href="#/login">Login</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#/register">Register</a>
                    </li>
                </ul>
                <ul class="navbar-nav ms-auto align-items-center" v-else>
                    <li class="nav-item me-3 d-flex align-items-center text-white">
                        <i class="bi bi-person-circle me-1"></i> {{ user.name }} &nbsp;<span class="badge bg-secondary">{{ user.role }}</span>
                    </li>
                    <li class="nav-item" v-if="user.role === 'admin'">
                        <a class="nav-link" href="#/admin">Dashboard</a>
                    </li>
                    <li class="nav-item" v-if="user.role === 'company'">
                        <a class="nav-link" href="#/company">Dashboard</a>
                    </li>
                    <li class="nav-item" v-if="user.role === 'student'">
                        <a class="nav-link" href="#/student">Dashboard</a>
                    </li>
                    <li class="nav-item ms-2">
                        <button class="btn btn-sm btn-outline-danger" @click="logout">
                            <i class="bi bi-box-arrow-right me-1"></i>Logout
                        </button>
                    </li>
                </ul>
            </div>
        </div>
    </nav>
    `,
    data() {
        return { user: null }
    },
    created() {
        this.checkUser();
        window.addEventListener('login-success', this.checkUser);
        window.addEventListener('logout-success', this.checkUser);
    },
    unmounted() {
        window.removeEventListener('login-success', this.checkUser);
        window.removeEventListener('logout-success', this.checkUser);
    },
    methods: {
        checkUser() {
            const userStr = localStorage.getItem('user');
            this.user = userStr ? JSON.parse(userStr) : null;
        },
        logout() {
            localStorage.removeItem('token');
            localStorage.removeItem('user');
            this.user = null;
            window.dispatchEvent(new Event('logout-success'));
            window.location.hash = '/login';
        }
    }
};
